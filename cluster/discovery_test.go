package cluster

import (
	"context"
	"encoding/json"
	"net"
	"slices"
	"testing"
	"time"

	"github.com/borism/ollama-cluster/ml"
)

// TestAnnouncementJSONRoundTrip covers the wire format without touching the
// network: encode an announcement the way broadcastLoop does, decode it the
// way listenLoop does, and check every field survives.
func TestAnnouncementJSONRoundTrip(t *testing.T) {
	want := announcement{
		ID:      "peer-1",
		Addrs:   []string{"192.168.1.10", "192.168.2.10"},
		RPCPort: 50052,
		Devices: []ml.DeviceInfo{
			{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "A4500", TotalMemory: 20 << 30, FreeMemory: 12 << 30},
		},
		ProtoMajor: RPCProtoMajor,
		ProtoMinor: RPCProtoMinor,
		Load:       0.42,
	}

	buf, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got announcement
	if err := json.Unmarshal(buf, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.ID != want.ID {
		t.Fatalf("ID: got %q want %q", got.ID, want.ID)
	}
	if got.RPCPort != want.RPCPort {
		t.Fatalf("RPCPort: got %d want %d", got.RPCPort, want.RPCPort)
	}
	if got.ProtoMajor != want.ProtoMajor || got.ProtoMinor != want.ProtoMinor {
		t.Fatalf("proto: got %d.%d want %d.%d", got.ProtoMajor, got.ProtoMinor, want.ProtoMajor, want.ProtoMinor)
	}
	if got.Load != want.Load {
		t.Fatalf("Load: got %v want %v", got.Load, want.Load)
	}
	if len(got.Devices) != 1 || got.Devices[0].Name != "A4500" || got.Devices[0].FreeMemory != 12<<30 {
		t.Fatalf("Devices: got %+v", got.Devices)
	}
	if !slices.Equal(got.Addrs, want.Addrs) {
		t.Fatalf("Addrs: got %v want %v", got.Addrs, want.Addrs)
	}
}

// TestTablePeersExpiry checks lazy expiry: a peer heard long enough ago
// (older than TTL) must not show up in Peers(), a fresh one must.
func TestTablePeersExpiry(t *testing.T) {
	tbl := &Table{
		peers: make(map[string]Peer),
		ttl:   time.Minute,
	}
	tbl.observe(Peer{ID: "stale", LastSeen: time.Now().Add(-time.Hour)})
	tbl.observe(Peer{ID: "fresh", LastSeen: time.Now()})

	peers := tbl.Peers()
	if len(peers) != 1 || peers[0].ID != "fresh" {
		t.Fatalf("Peers() = %+v, want only \"fresh\"", peers)
	}
}

// TestTableSelf checks Self() reflects the most recent updateSelf call.
func TestTableSelf(t *testing.T) {
	tbl := &Table{peers: make(map[string]Peer)}
	tbl.updateSelf(Peer{ID: "me", RPCPort: 1234})
	if got := tbl.Self(); got.ID != "me" || got.RPCPort != 1234 {
		t.Fatalf("Self() = %+v", got)
	}
}

// TestListenLoopReceivesAnnouncement is the one real-networking test: it
// exercises Start's actual listening socket over loopback UDP unicast
// rather than OS broadcast, since a broadcast to 255.255.255.255 typically
// isn't delivered back to a 127.0.0.1 listener (and there's no second host
// to broadcast to in this sandbox). A hard context timeout means a genuine
// networking problem here fails the test instead of hanging.
func TestListenLoopReceivesAnnouncement(t *testing.T) {
	// Grab a free UDP port from the OS, then release it for Start to reuse.
	probe, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Skipf("no loopback UDP available in this sandbox: %v", err)
	}
	port := probe.LocalAddr().(*net.UDPAddr).Port
	probe.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cfg := Config{
		Port:        port,
		Interval:    time.Hour, // don't let our own broadcastLoop interfere
		TTL:         time.Minute,
		SelfID:      "self",
		SelfDevices: func() []ml.DeviceInfo { return nil },
		SelfRPCPort: func() int { return 0 },
		SelfLoad:    func() float64 { return 0 },
	}
	tbl, err := Start(ctx, cfg)
	if err != nil {
		t.Skipf("could not start discovery listener in this sandbox: %v", err)
	}

	sender, err := net.DialUDP("udp4", nil, &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: port})
	if err != nil {
		t.Skipf("no loopback UDP send available in this sandbox: %v", err)
	}
	defer sender.Close()

	buf, err := json.Marshal(announcement{
		ID:         "peer-2",
		RPCPort:    50052,
		ProtoMajor: RPCProtoMajor,
		ProtoMinor: RPCProtoMinor,
		Load:       0.1,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	deadline := time.Now().Add(4 * time.Second)
	for {
		if _, err := sender.Write(buf); err != nil {
			t.Fatalf("send: %v", err)
		}
		for _, p := range tbl.Peers() {
			if p.ID == "peer-2" {
				return // found it, test passes
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("peer-2 never appeared in the table within the timeout")
		}
		select {
		case <-ctx.Done():
			t.Fatal("context timed out waiting for peer-2")
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// TestObservePreservesProbedAddr guards the fix for a multi-homed peer
// getting pinned to whichever interface's beacon last happened to arrive:
// once probeLoop has picked a best address via setBest, a routine new
// beacon (observe) must not clobber it back to the beacon's own source IP.
func TestObservePreservesProbedAddr(t *testing.T) {
	tbl := &Table{peers: make(map[string]Peer)}
	tbl.observe(Peer{ID: "p1", Addr: "10.0.0.1", Addrs: []string{"10.0.0.1", "10.0.0.2"}, LastSeen: time.Now()})
	tbl.setBest("p1", "10.0.0.2", 2*time.Millisecond)

	// A fresh beacon arrives from the slower address again.
	tbl.observe(Peer{ID: "p1", Addr: "10.0.0.1", Addrs: []string{"10.0.0.1", "10.0.0.2"}, LastSeen: time.Now()})

	got := tbl.peers["p1"]
	if got.Addr != "10.0.0.2" {
		t.Fatalf("Addr = %q, want probeLoop's pick 10.0.0.2 to survive the new beacon", got.Addr)
	}
	if got.Latency != 2*time.Millisecond {
		t.Fatalf("Latency = %v, want the probed value to survive the new beacon", got.Latency)
	}
}

// TestProbeOncePicksReachableCandidate is the one real-networking test for
// multi-address selection: a peer with two candidate addresses, one with
// nothing listening, must end up pinned to the one that actually answers.
func TestProbeOncePicksReachableCandidate(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("no loopback TCP available in this sandbox: %v", err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	port := ln.Addr().(*net.TCPAddr).Port

	// TEST-NET-1 (RFC 5737, 192.0.2.0/24): guaranteed non-routable, so this
	// dial fails fast (or times out) rather than actually connecting.
	tbl := &Table{peers: map[string]Peer{
		"p1": {
			ID:       "p1",
			Addr:     "192.0.2.1", // wrong pick, as if the last beacon came in on the dead interface
			Addrs:    []string{"192.0.2.1", "127.0.0.1"},
			RPCPort:  port,
			LastSeen: time.Now(),
		},
	}, ttl: time.Minute}

	probeOnce(tbl)

	got := tbl.peers["p1"]
	if got.Addr != "127.0.0.1" {
		t.Fatalf("Addr = %q, want probeOnce to have corrected it to the reachable candidate 127.0.0.1", got.Addr)
	}
	if got.Latency <= 0 {
		t.Fatalf("Latency = %v, want a positive measured value", got.Latency)
	}
}

// TestProbePeerDialsCandidatesConcurrently guards against the obvious way
// to re-break this: two unreachable candidates plus one reachable one.
// Dialed one-by-one this would take at least 2*probeTimeout; dialed
// concurrently, about one probeTimeout regardless of candidate count.
func TestProbePeerDialsCandidatesConcurrently(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("no loopback TCP available in this sandbox: %v", err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	port := ln.Addr().(*net.TCPAddr).Port

	p := Peer{
		ID:       "p1",
		Addrs:    []string{"192.0.2.1", "192.0.2.2", "127.0.0.1"}, // 192.0.2.0/24 (RFC 5737) is non-routable
		RPCPort:  port,
		LastSeen: time.Now(),
	}
	tbl := &Table{peers: map[string]Peer{"p1": p}}

	start := time.Now()
	probePeer(tbl, p)
	elapsed := time.Since(start)

	if elapsed > probeTimeout+250*time.Millisecond {
		t.Fatalf("probePeer took %v, want close to one probeTimeout (%v) -- candidates aren't being dialed concurrently", elapsed, probeTimeout)
	}
	if got := tbl.peers["p1"].Addr; got != "127.0.0.1" {
		t.Fatalf("Addr = %q, want the reachable candidate", got)
	}
}

// TestBackoffDelayIncreasesThenCaps checks the doubling schedule and cap
// directly, without waiting real time out.
func TestBackoffDelayIncreasesThenCaps(t *testing.T) {
	cases := []struct {
		failures int
		want     time.Duration
	}{
		{1, probeInterval},
		{2, 2 * probeInterval},
		{3, 4 * probeInterval},
	}
	for _, c := range cases {
		if got := backoffDelay(c.failures); got != c.want {
			t.Errorf("backoffDelay(%d) = %v, want %v", c.failures, got, c.want)
		}
	}
	if got := backoffDelay(20); got != probeBackoffMax {
		t.Errorf("backoffDelay(20) = %v, want cap %v", got, probeBackoffMax)
	}
}

// TestShouldProbeBacksOffThenRecovers checks the Table bookkeeping
// backoffDelay feeds: a failure makes the candidate not due, a success
// clears that immediately.
func TestShouldProbeBacksOffThenRecovers(t *testing.T) {
	tbl := &Table{peers: make(map[string]Peer)}
	key := "p1|10.0.0.9"

	if !tbl.shouldProbe(key) {
		t.Fatal("a never-probed candidate should be due")
	}
	tbl.recordProbeFailure(key)
	if tbl.shouldProbe(key) {
		t.Fatal("immediately after a failure, the candidate should be backed off")
	}
	tbl.recordProbeSuccess(key)
	if !tbl.shouldProbe(key) {
		t.Fatal("a success should clear backoff immediately")
	}
}

// TestProbePeerSkipsBackedOffCandidate is the end-to-end version: a
// candidate with an established failure streak must not be redialed at
// all (not just deprioritized), even though it's still listed in Addrs.
func TestProbePeerSkipsBackedOffCandidate(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("no loopback TCP available in this sandbox: %v", err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	port := ln.Addr().(*net.TCPAddr).Port

	p := Peer{ID: "p1", Addrs: []string{"192.0.2.1", "127.0.0.1"}, RPCPort: port, LastSeen: time.Now()}
	tbl := &Table{peers: map[string]Peer{"p1": p}}

	// Give the unreachable candidate an established failure streak so
	// it's now well into its backoff window.
	tbl.recordProbeFailure("p1|192.0.2.1")
	tbl.recordProbeFailure("p1|192.0.2.1")

	start := time.Now()
	probePeer(tbl, p)
	elapsed := time.Since(start)

	// If the backed-off candidate were dialed anyway, this would take
	// close to probeTimeout; skipping it should be near-instant.
	if elapsed > 250*time.Millisecond {
		t.Fatalf("probePeer took %v, want near-instant with the unreachable candidate backed off", elapsed)
	}
	if got := tbl.peers["p1"].Addr; got != "127.0.0.1" {
		t.Fatalf("Addr = %q, want the reachable candidate", got)
	}
}

func TestResolveSeeds(t *testing.T) {
	addrs := resolveSeeds([]string{"127.0.0.1:11435", "not a valid seed:::", "192.168.1.42:50999"})
	if len(addrs) != 2 {
		t.Fatalf("expected 2 resolved seeds (1 invalid skipped), got %d: %v", len(addrs), addrs)
	}
	if addrs[0].String() != "127.0.0.1:11435" {
		t.Errorf("addrs[0] = %v, want 127.0.0.1:11435", addrs[0])
	}
	if addrs[1].String() != "192.168.1.42:50999" {
		t.Errorf("addrs[1] = %v, want 192.168.1.42:50999", addrs[1])
	}
}

func TestResolveSeedsEmpty(t *testing.T) {
	if addrs := resolveSeeds(nil); len(addrs) != 0 {
		t.Errorf("resolveSeeds(nil) = %v, want empty", addrs)
	}
}

// TestStartStoppedReleasesPort: a cluster restart (server/cluster.go)
// cancels discovery, waits on Stopped, and binds the same UDP port again
// straight away.
func TestStartStoppedReleasesPort(t *testing.T) {
	probe, err := net.ListenUDP("udp4", &net.UDPAddr{})
	if err != nil {
		t.Fatal(err)
	}
	port := probe.LocalAddr().(*net.UDPAddr).Port
	probe.Close()

	cfg := Config{
		Port:        port,
		Interval:    time.Hour,
		TTL:         time.Hour,
		SelfDevices: func() []ml.DeviceInfo { return nil },
		SelfRPCPort: func() int { return 0 },
		SelfLoad:    func() float64 { return 0 },
	}
	ctx, cancel := context.WithCancel(t.Context())
	table, err := Start(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-table.Stopped():
	case <-time.After(5 * time.Second):
		t.Fatal("Stopped not closed after cancel")
	}

	ctx2, cancel2 := context.WithCancel(t.Context())
	defer cancel2()
	if _, err := Start(ctx2, cfg); err != nil {
		t.Fatalf("restart on the same port: %v", err)
	}
}
