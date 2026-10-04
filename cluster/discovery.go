package cluster

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/borism/ollama-cluster/ml"
)

// RPCProtoMajor/RPCProtoMinor are the RPC_PROTO_MAJOR_VERSION/
// RPC_PROTO_MINOR_VERSION this build's vendored llama.cpp speaks
// (ggml/include/ggml-rpc.h at the LLAMA_CPP_VERSION tag). Bump these
// whenever LLAMA_CPP_VERSION changes the RPC protocol: peers advertise
// them, and a stale value pairs incompatible builds that then fail the
// RPC handshake at model load. 7.0.0 since b11081 ("rpc : hash-cache only
// weights").
const (
	RPCProtoMajor = 7
	RPCProtoMinor = 0
)

// announcement is the JSON payload broadcast on the wire. It's the subset
// of Peer this node can self-report; Addr and LastSeen are filled in by the
// receiver, not the sender.
type announcement struct {
	ID         string          `json:"id"`
	Addrs      []string        `json:"addrs,omitempty"`
	RPCPort    int             `json:"rpc_port"`
	Devices    []ml.DeviceInfo `json:"devices"`
	ProtoMajor int             `json:"proto_major"`
	ProtoMinor int             `json:"proto_minor"`
	Load       float64         `json:"load"`
}

// Table is a thread-safe, live set of peers seen via discovery beacons.
type Table struct {
	mu      sync.Mutex
	self    Peer
	peers   map[string]Peer
	ttl     time.Duration
	stopped chan struct{}

	// backoff tracks consecutive probe failures per "peerID|addr"
	// candidate (see shouldProbe/recordProbeFailure), so a candidate
	// that's never reachable from here -- e.g. a peer's Thunderbolt
	// bridge address, which only the *other* end of that cable can ever
	// dial -- stops being redialed every probeInterval forever.
	backoff map[string]probeBackoff
}

// Stopped is closed once discovery has shut down after its context was
// canceled and released its UDP port, so a restart can bind it again.
func (t *Table) Stopped() <-chan struct{} {
	return t.stopped
}

// Self returns our own current advertised state, for debugging/logging.
func (t *Table) Self() Peer {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.self
}

// Peers returns live peers, excluding stale ones per Config.TTL and
// excluding our own ID.
func (t *Table) Peers() []Peer {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now()
	peers := make([]Peer, 0, len(t.peers))
	for _, p := range t.peers {
		if !p.Stale(now, t.ttl) {
			peers = append(peers, p)
		}
	}
	return peers
}

func (t *Table) updateSelf(p Peer) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.self = p
}

func (t *Table) observe(p Peer) {
	t.mu.Lock()
	defer t.mu.Unlock()
	// A fresh beacon only refreshes identity/capacity fields. Which of
	// Addrs is actually fastest is probeLoop's job, on its own slower
	// cadence -- don't let every 5s beacon stomp the address it already
	// worked out, back to a merely-provisional one. BandwidthMbps is the
	// same story one level further out (bandwidthLoop's 5-minute cadence):
	// an announcement never carries it at all, so without this a beacon
	// arriving even a second after a real measurement wiped it straight
	// back to zero -- the bug that made bandwidth look permanently
	// unmeasured despite bandwidthPeer succeeding.
	if existing, ok := t.peers[p.ID]; ok && existing.Addr != "" {
		p.Addr = existing.Addr
		p.Latency = existing.Latency
		p.BandwidthMbps = existing.BandwidthMbps
	}
	t.peers[p.ID] = p
}

// probeBackoff is one candidate address's consecutive-failure streak (see
// Table.backoff).
type probeBackoff struct {
	failures int
	nextTry  time.Time
}

// probeBackoffMax caps how long a failing candidate is skipped before
// being retried, so a genuinely-recovered link (cable replugged, firewall
// reopened) doesn't stay out of rotation forever.
const probeBackoffMax = 5 * time.Minute

// backoffDelay is how long to wait before redialing a candidate that has
// now failed `failures` times in a row: one failure gets no extra delay
// (transient blips are common and cheap to just retry next cycle), each
// further consecutive failure doubles the wait, up to probeBackoffMax.
func backoffDelay(failures int) time.Duration {
	d := probeInterval
	for i := 1; i < failures && d < probeBackoffMax; i++ {
		d *= 2
	}
	if d > probeBackoffMax {
		d = probeBackoffMax
	}
	return d
}

// shouldProbe reports whether a "peerID|addr" candidate is due for a
// fresh dial, or still serving out backoff from previous failures.
func (t *Table) shouldProbe(key string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	bo, ok := t.backoff[key]
	return !ok || !time.Now().Before(bo.nextTry)
}

// recordProbeSuccess clears any backoff for key: a candidate that just
// answered is trusted again immediately, not eased back in gradually.
func (t *Table) recordProbeSuccess(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.backoff, key)
}

// recordProbeFailure bumps key's failure streak and schedules its next
// retry per backoffDelay.
func (t *Table) recordProbeFailure(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.backoff == nil {
		t.backoff = make(map[string]probeBackoff)
	}
	bo := t.backoff[key]
	bo.failures++
	bo.nextTry = time.Now().Add(backoffDelay(bo.failures))
	t.backoff[key] = bo
}

// setBest records the fastest of a peer's candidate addresses (see
// probeLoop). A no-op if the peer expired between listing and probing --
// nothing to update.
func (t *Table) setBest(id, addr string, d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.peers[id]; ok {
		p.Addr = addr
		p.Latency = d
		t.peers[id] = p
	}
}

// randomID returns a random hex string for use as a self ID, for
// convenience when Config.SelfID is left empty.
func randomID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand.Read on any real platform doesn't fail; fall back
		// to something unique-ish rather than erroring Start out.
		binary.BigEndian.PutUint64(b[:], uint64(time.Now().UnixNano()))
	}
	return fmt.Sprintf("%x", b)
}

// Start launches the discovery beacon: one goroutine periodically
// broadcasts our own state over UDP, one listens for peers' broadcasts and
// updates the returned Table. Both stop when ctx is canceled.
func Start(ctx context.Context, cfg Config) (*Table, error) {
	if cfg.SelfID == "" {
		cfg.SelfID = randomID()
	}

	conn, err := net.ListenUDP("udp4", &net.UDPAddr{Port: cfg.Port})
	if err != nil {
		return nil, fmt.Errorf("cluster: listen udp :%d: %w", cfg.Port, err)
	}
	bwLn, err := bandwidthListen(cfg.Port)
	if err != nil {
		conn.Close()
		return nil, err
	}

	t := &Table{
		peers:   make(map[string]Peer),
		ttl:     cfg.TTL,
		stopped: make(chan struct{}),
	}

	go broadcastLoop(ctx, conn, cfg, t)
	go listenLoop(ctx, conn, cfg, t)
	go probeLoop(ctx, t)
	go bandwidthLoop(ctx, t, cfg.Port)
	go acceptBandwidthConns(ctx, bwLn)

	go func() {
		<-ctx.Done()
		conn.Close()
		bwLn.Close()
		close(t.stopped)
	}()

	return t, nil
}

func selfAnnouncement(cfg Config) announcement {
	return announcement{
		ID:         cfg.SelfID,
		Addrs:      selfAddrs(),
		RPCPort:    cfg.SelfRPCPort(),
		Devices:    cfg.SelfDevices(),
		ProtoMajor: RPCProtoMajor,
		ProtoMinor: RPCProtoMinor,
		Load:       cfg.SelfLoad(),
	}
}

// selfAddrs returns this host's own IPv4 addresses across every up,
// non-loopback interface, self-reported in the announcement as Peer.Addrs.
// A multi-homed host (e.g. a laptop with both Ethernet and Wi-Fi up) lists
// more than one; the receiver's probeLoop measures each and keeps the
// fastest, rather than whichever address happened to carry a given beacon
// (see Peer.Addr).
func selfAddrs() []string {
	var out []string
	ifaces, err := net.Interfaces()
	if err != nil {
		return out
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			ipnet, ok := a.(*net.IPNet)
			if !ok {
				continue
			}
			if ip4 := ipnet.IP.To4(); ip4 != nil {
				out = append(out, ip4.String())
			}
		}
	}
	return out
}

// broadcastAddrs returns the subnet-directed broadcast address (e.g.
// 192.168.1.255) of every up, non-loopback IPv4 interface, plus the global
// limited-broadcast address 255.255.255.255 as a best-effort fallback.
// Confirmed by hand against a real network: some LANs/routers deliver
// subnet-directed broadcast but silently drop 255.255.255.255, so that alone isn't enough
// -- and a multi-homed host (more than one NIC on the LAN) needs its own
// address computed per interface, not just the default route's.
func broadcastAddrs(port int) []*net.UDPAddr {
	addrs := []*net.UDPAddr{{IP: net.IPv4bcast, Port: port}}

	ifaces, err := net.Interfaces()
	if err != nil {
		return addrs
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 || iface.Flags&net.FlagBroadcast == 0 {
			continue
		}
		ifaceAddrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range ifaceAddrs {
			ipnet, ok := a.(*net.IPNet)
			if !ok {
				continue
			}
			ip4 := ipnet.IP.To4()
			if ip4 == nil {
				continue
			}
			bcast := make(net.IP, 4)
			for i := range ip4 {
				bcast[i] = ip4[i] | ^ipnet.Mask[i]
			}
			addrs = append(addrs, &net.UDPAddr{IP: bcast, Port: port})
		}
	}
	return addrs
}

// resolveSeeds resolves Config.Seeds ("host:port" strings) to UDP
// addresses, logging and skipping any that fail to resolve rather than
// failing Start outright -- a typo'd or momentarily-unreachable seed
// shouldn't take down LAN broadcast discovery too.
func resolveSeeds(seeds []string) []*net.UDPAddr {
	addrs := make([]*net.UDPAddr, 0, len(seeds))
	for _, s := range seeds {
		addr, err := net.ResolveUDPAddr("udp4", s)
		if err != nil {
			slog.Warn("cluster: could not resolve seed, skipping", "seed", s, "error", err)
			continue
		}
		addrs = append(addrs, addr)
	}
	return addrs
}

// broadcastLoop periodically encodes and sends our own announcement.
func broadcastLoop(ctx context.Context, conn *net.UDPConn, cfg Config, t *Table) {
	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()

	send := func() {
		// Recomputed on every tick, not cached: a laptop that switches
		// interfaces (Ethernet -> Wi-Fi, or gets a new address on wake)
		// needs its next beacon to go out the current interface list, not
		// whatever was up when the loop started.
		dsts := append(broadcastAddrs(cfg.Port), resolveSeeds(cfg.Seeds)...)

		a := selfAnnouncement(cfg)
		t.updateSelf(Peer{
			ID:         a.ID,
			RPCPort:    a.RPCPort,
			Devices:    a.Devices,
			ProtoMajor: a.ProtoMajor,
			ProtoMinor: a.ProtoMinor,
			Load:       a.Load,
			LastSeen:   time.Now(),
		})
		buf, err := json.Marshal(a)
		if err != nil {
			slog.Warn("cluster: encode announcement", "error", err)
			return
		}
		for _, dst := range dsts {
			if _, err := conn.WriteToUDP(buf, dst); err != nil {
				slog.Debug("cluster: broadcast announcement", "dst", dst, "error", err)
			}
		}
	}

	send()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			send()
		}
	}
}

// probeInterval/probeTimeout govern probeLoop's link-latency measurement
// to known peers, consumed by placement.go's water-fill.
const (
	probeInterval = 10 * time.Second
	probeTimeout  = 500 * time.Millisecond
)

// probeLoop periodically times a TCP dial to each of a live peer's
// candidate RPC addresses (Peer.Addrs) and keeps the fastest as Peer.Addr.
// This is a real measurement, not a placeholder: the dial's handshake is
// one round trip over the same link the RPC traffic itself would use. A
// multi-homed peer (e.g. a laptop with both Ethernet and Wi-Fi up) can
// otherwise end up pinned to whichever interface happened to carry its
// most recent discovery beacon -- an accident of routing, not a measured
// choice, and one link can be several times slower than the other. A peer
// with none of its candidates reachable this round just keeps its
// last-known address/latency (or zero, if never probed) -- water-fill
// treats that as "no measured cost yet" rather than excluding the peer.
func probeLoop(ctx context.Context, t *Table) {
	ticker := time.NewTicker(probeInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			probeOnce(t)
		}
	}
}

// probeOnce is one probeLoop pass, pulled out so tests can trigger it
// directly instead of waiting on probeInterval's real-time ticker. Peers,
// and each peer's own candidates, are all dialed concurrently -- an
// unreachable candidate costs at most probeTimeout, never probeTimeout
// times the number of peers times the number of candidates, however many
// interfaces the fleet grows to report.
func probeOnce(t *Table) {
	var wg sync.WaitGroup
	for _, p := range t.Peers() {
		if p.RPCPort == 0 {
			continue
		}
		wg.Add(1)
		go func(p Peer) {
			defer wg.Done()
			probePeer(t, p)
		}(p)
	}
	wg.Wait()
}

// probePeer dials every one of p's candidate addresses concurrently,
// skipping any still backed off (see Table.shouldProbe), and records the
// fastest that answers as the peer's new best address. A candidate that's
// never reachable from here -- a peer's Thunderbolt-bridge link-local
// address, say, which only the other end of that cable can dial -- stops
// being retried every cycle once it's built up a failure streak.
func probePeer(t *Table, p Peer) {
	candidates := p.Addrs
	if len(candidates) == 0 {
		candidates = []string{p.Addr}
	}

	type probed struct {
		addr    string
		latency time.Duration
	}
	results := make(chan probed, len(candidates))

	var wg sync.WaitGroup
	due := 0
	for _, ip := range candidates {
		key := p.ID + "|" + ip
		if !t.shouldProbe(key) {
			continue
		}
		due++
		wg.Add(1)
		go func(ip, key string) {
			defer wg.Done()
			addr := net.JoinHostPort(ip, strconv.Itoa(p.RPCPort))
			start := time.Now()
			conn, err := net.DialTimeout("tcp", addr, probeTimeout)
			if err != nil {
				slog.Debug("cluster: latency probe failed, backing off", "id", p.ID, "addr", addr, "error", err)
				t.recordProbeFailure(key)
				return
			}
			conn.Close()
			t.recordProbeSuccess(key)
			results <- probed{ip, time.Since(start)}
		}(ip, key)
	}
	if due == 0 {
		return // every candidate is still serving out backoff
	}
	wg.Wait()
	close(results)

	var bestAddr string
	var bestLatency time.Duration
	for r := range results {
		if bestAddr == "" || r.latency < bestLatency {
			bestAddr, bestLatency = r.addr, r.latency
		}
	}
	if bestAddr == "" {
		return // every due candidate failed, keep the last-known address
	}
	slog.Debug("cluster: probed peer latency", "id", p.ID, "addr", bestAddr, "latency", bestLatency)
	t.setBest(p.ID, bestAddr, bestLatency)
}

// listenLoop reads incoming beacons and updates the table, ignoring our own.
func listenLoop(ctx context.Context, conn *net.UDPConn, cfg Config, t *Table) {
	buf := make([]byte, 64*1024)
	for {
		n, src, err := conn.ReadFromUDP(buf)
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				slog.Debug("cluster: read udp", "error", err)
				return
			}
		}
		var a announcement
		if err := json.Unmarshal(buf[:n], &a); err != nil {
			slog.Debug("cluster: decode announcement", "error", err, "src", src)
			continue
		}
		slog.Debug("cluster: received announcement", "id", a.ID, "self_id", cfg.SelfID, "src", src)
		if a.ID == "" || a.ID == cfg.SelfID {
			continue
		}
		srcAddr := src.IP.String()
		addrs := a.Addrs
		if !slices.Contains(addrs, srcAddr) {
			// Always include the address the beacon actually arrived from,
			// even if it's missing or stale in the peer's self-report --
			// it's proof that address works.
			addrs = append(addrs, srcAddr)
		}
		t.observe(Peer{
			ID:         a.ID,
			Addr:       srcAddr, // provisional; probeLoop picks the fastest of Addrs
			Addrs:      addrs,
			RPCPort:    a.RPCPort,
			Devices:    a.Devices,
			ProtoMajor: a.ProtoMajor,
			ProtoMinor: a.ProtoMinor,
			Load:       a.Load,
			LastSeen:   time.Now(),
		})
	}
}

// NewStaticTable returns a Table holding exactly peers, without starting
// discovery. For tests of code that consumes a Table.
func NewStaticTable(peers ...Peer) *Table {
	t := &Table{peers: map[string]Peer{}, ttl: time.Hour, stopped: make(chan struct{})}
	for _, p := range peers {
		p.LastSeen = time.Now()
		t.peers[p.ID] = p
	}
	return t
}
