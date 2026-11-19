package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"strconv"
	"sync"
	"time"
)

// bandwidthDuration is how long one bandwidth probe keeps pushing data at
// a peer's link. Time-based rather than a fixed payload so it can fill a
// fast link (a 2.5 Gbit link moves 4 MiB in ~13 ms, mostly TCP slow-start,
// which understated it badly) while a slow one still costs only this long.
// A var so tests can shorten it. Tunable; not yet measured across link
// speeds on real hardware.
var bandwidthDuration = 500 * time.Millisecond

// bandwidthChunk is the size of one write while pushing data, small enough
// that the duration check is honored within a chunk even on a slow link.
const bandwidthChunk = 256 * 1024

// bandwidthInterval is how often each peer's link is re-measured. Far
// longer than probeInterval (latency, a handshake-only dial) on purpose:
// this actually pushes bytes over the link, so running it that often would
// be exactly the "waste and delays" real workloads can't afford.
const bandwidthInterval = 5 * time.Minute

// bandwidthTimeout bounds one bandwidth probe end to end (dial, send,
// reply): bandwidthDuration plus draining whatever the link still has
// buffered. 10s keeps one struggling peer from stalling past its own probe
// (peers are measured concurrently, see bandwidthOnce, but each still needs
// its own ceiling).
const bandwidthTimeout = 10 * time.Second

// bandwidthResult is what the receiving end reports back over the same
// connection once it has read the whole payload: how many bytes arrived
// and how long that took, measured on the receiving side -- the side that
// actually experiences the receive window, and the direction weights flow
// in real RPC traffic (head to worker) -- not the sender's write-buffered
// view of its own throughput.
type bandwidthResult struct {
	Bytes int64 `json:"bytes"`
	Nanos int64 `json:"nanos"`
}

// bandwidthListen opens the bandwidth-probe listener. It shares
// Config.Port with the UDP discovery socket -- TCP and UDP are independent
// namespaces for the same port number, so this needs no port of its own to
// configure or document. The caller owns ln's lifecycle (see Start): it
// must be closed alongside the UDP socket, in the same place, so a
// restart on the same port can't race ahead of this one closing.
func bandwidthListen(port int) (net.Listener, error) {
	ln, err := net.Listen("tcp4", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, fmt.Errorf("cluster: listen tcp :%d: %w", port, err)
	}
	return ln, nil
}

func acceptBandwidthConns(ctx context.Context, ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				slog.Debug("cluster: bandwidth accept", "error", err)
				return
			}
		}
		go serveBandwidthConn(conn)
	}
}

// serveBandwidthConn reads one probe's payload to EOF -- the client
// half-closes its write side once bandwidthDuration is up -- and reports
// back how much arrived and how long that took.
func serveBandwidthConn(conn net.Conn) {
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(bandwidthTimeout))
	start := time.Now()
	n, err := io.Copy(io.Discard, conn)
	if err != nil {
		slog.Debug("cluster: bandwidth serve", "error", err)
		return
	}
	result := bandwidthResult{Bytes: n, Nanos: time.Since(start).Nanoseconds()}
	json.NewEncoder(conn).Encode(result)
}

// bandwidthLoop periodically re-measures every peer's link throughput, on
// its own slower cadence (see bandwidthInterval).
func bandwidthLoop(ctx context.Context, t *Table, port int) {
	ticker := time.NewTicker(bandwidthInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bandwidthOnce(t, port)
		}
	}
}

// bandwidthOnce measures every peer's currently-best address (see
// Peer.Addr) concurrently -- the one probeLoop already established has the
// lowest latency, not every candidate interface again, which would double
// the bytes pushed over the LAN for no placement-relevant benefit.
func bandwidthOnce(t *Table, port int) {
	var wg sync.WaitGroup
	for _, p := range t.Peers() {
		if p.Addr == "" {
			continue
		}
		wg.Add(1)
		go func(p Peer) {
			defer wg.Done()
			bandwidthPeer(t, p, port)
		}(p)
	}
	wg.Wait()
}

// bandwidthPeer measures p's current best address and records the result
// as the peer's BandwidthMbps. A failure just leaves the peer's last-known
// BandwidthMbps in place (or zero, if never measured) and is retried next
// bandwidthInterval.
//
// ponytail: no backoff here, unlike probePeer's per-candidate one -- this
// already only runs once every bandwidthInterval per peer, so there's no
// hot loop to protect against. Add one if a chronically-unreachable peer's
// bandwidth port turns out to be a nuisance in practice.
func bandwidthPeer(t *Table, p Peer, port int) {
	mbps, err := measureBandwidth(p.Addr, port)
	if err != nil {
		slog.Debug("cluster: bandwidth probe failed", "id", p.ID, "addr", p.Addr, "error", err)
		return
	}
	slog.Debug("cluster: probed peer bandwidth", "id", p.ID, "addr", p.Addr, "mbps", mbps)
	t.setLinkBandwidth(p.ID, p.Addr, mbps)
	t.setBandwidth(p.ID, mbps)
}

// measureBandwidth pushes zeros at ip's bandwidth port for
// bandwidthDuration and returns the throughput the far end reports back,
// in megabits/sec.
func measureBandwidth(ip string, port int) (float64, error) {
	conn, err := net.DialTimeout("tcp", net.JoinHostPort(ip, strconv.Itoa(port)), bandwidthTimeout)
	if err != nil {
		return 0, err
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(bandwidthTimeout))

	buf := make([]byte, bandwidthChunk)
	for end := time.Now().Add(bandwidthDuration); time.Now().Before(end); {
		if _, err := conn.Write(buf); err != nil {
			return 0, err
		}
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.CloseWrite()
	}

	var result bandwidthResult
	if err := json.NewDecoder(conn).Decode(&result); err != nil {
		return 0, err
	}
	if result.Nanos <= 0 {
		return 0, fmt.Errorf("cluster: bandwidth reply has no duration")
	}
	return float64(result.Bytes*8) / (float64(result.Nanos) / 1e9) / 1e6, nil
}

// linkStat is one candidate address's last bandwidth measurement.
type linkStat struct {
	mbps float64
	at   time.Time
}

// setLinkBandwidth records a measurement for one "peerID|addr" candidate,
// so probePeer can compare candidates without re-measuring every 10s.
func (t *Table) setLinkBandwidth(id, ip string, mbps float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.links == nil {
		t.links = make(map[string]linkStat)
	}
	t.links[id+"|"+ip] = linkStat{mbps, time.Now()}
}

// linkBandwidth returns a candidate's measurement if it is younger than
// bandwidthInterval, else 0.
func (t *Table) linkBandwidth(id, ip string) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	if s, ok := t.links[id+"|"+ip]; ok && time.Since(s.at) < bandwidthInterval {
		return s.mbps
	}
	return 0
}

// setBandwidth records a peer's last-measured link throughput. A no-op if
// the peer expired between listing and measuring, same as setBest.
func (t *Table) setBandwidth(id string, mbps float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.peers[id]; ok {
		p.BandwidthMbps = mbps
		t.peers[id] = p
	}
}
