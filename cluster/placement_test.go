package cluster

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"log/slog"
	"math"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/ml"
)

func gpu(freeMB uint64) ml.DeviceInfo {
	return ml.DeviceInfo{FreeMemory: freeMB * 1024 * 1024}
}

func peer(addr string, rpcPort int, protoMajor, protoMinor int, freeMB uint64) Peer {
	return Peer{
		Addr:       addr,
		RPCPort:    rpcPort,
		ProtoMajor: protoMajor,
		ProtoMinor: protoMinor,
		Devices:    []ml.DeviceInfo{gpu(freeMB)},
	}
}

func peerAt(addr string, freeMB uint64, load float64, latency time.Duration) Peer {
	p := peer(addr, 50052, RPCProtoMajor, 0, freeMB)
	p.Load = load
	p.Latency = latency
	return p
}

func mib(n uint64) uint64 { return n * 1024 * 1024 }

func TestSelectRPCServers(t *testing.T) {
	cases := []struct {
		name     string
		gpus     []ml.DeviceInfo
		predict  uint64
		peers    []Peer
		opts     api.Options
		wantRPCs string
	}{
		{
			name:     "local already fits, no-op",
			gpus:     []ml.DeviceInfo{gpu(8000)},
			predict:  mib(4000),
			peers:    []Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 20000)},
			wantRPCs: "",
		},
		{
			name:     "one peer covers the shortfall",
			gpus:     []ml.DeviceInfo{gpu(2000)},
			predict:  mib(4000),
			peers:    []Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 20000)},
			wantRPCs: "10.0.0.2:50052",
		},
		{
			name:    "multiple peers needed",
			gpus:    []ml.DeviceInfo{gpu(1000)},
			predict: mib(10000),
			peers: []Peer{
				peer("10.0.0.2", 50052, RPCProtoMajor, 0, 6000),
				peer("10.0.0.3", 50052, RPCProtoMajor, 0, 5000),
				peer("10.0.0.4", 50052, RPCProtoMajor, 0, 1000),
			},
			// local available = 1000 MiB - 457 MiB MinimumMemory reserve = 543 MiB.
			// sorted descending by free memory: .2 (6000), .3 (5000), .4 (1000)
			// 543 + 6000 = 6543 < 10000, keep going
			// 6543 + 5000 = 11543 >= 10000, stop before adding .4
			wantRPCs: "10.0.0.2:50052,10.0.0.3:50052",
		},
		{
			name:    "wrong ProtoMajor is skipped",
			gpus:    []ml.DeviceInfo{gpu(1000)},
			predict: mib(10000),
			peers: []Peer{
				peer("10.0.0.2", 50052, RPCProtoMajor+1, 0, 50000),
				peer("10.0.0.3", 50052, RPCProtoMajor, 0, 9000),
			},
			wantRPCs: "10.0.0.3:50052",
		},
		{
			name:    "peer ProtoMinor above ours is skipped",
			gpus:    []ml.DeviceInfo{gpu(1000)},
			predict: mib(10000),
			peers: []Peer{
				peer("10.0.0.2", 50052, RPCProtoMajor, RPCProtoMinor+1, 50000),
				peer("10.0.0.3", 50052, RPCProtoMajor, 0, 9000),
			},
			wantRPCs: "10.0.0.3:50052",
		},
		{
			name:    "RPCPort zero is skipped",
			gpus:    []ml.DeviceInfo{gpu(1000)},
			predict: mib(10000),
			peers: []Peer{
				peer("10.0.0.2", 0, RPCProtoMajor, 0, 50000),
				peer("10.0.0.3", 50052, RPCProtoMajor, 0, 9000),
			},
			wantRPCs: "10.0.0.3:50052",
		},
		{
			name:     "explicit RPCServers already set is left untouched",
			gpus:     []ml.DeviceInfo{gpu(100)},
			predict:  mib(10000),
			peers:    []Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 50000)},
			opts:     api.Options{Runner: api.Runner{RPCServers: "1.2.3.4:9999"}},
			wantRPCs: "1.2.3.4:9999",
		},
		{
			name:     "zero usable peers leaves RPCServers empty",
			gpus:     []ml.DeviceInfo{gpu(100)},
			predict:  mib(10000),
			peers:    []Peer{peer("10.0.0.2", 0, RPCProtoMajor, 0, 50000)},
			wantRPCs: "",
		},
		{
			name:    "partial help still added when no peer covers it fully",
			gpus:    []ml.DeviceInfo{gpu(100)},
			predict: mib(100000),
			peers: []Peer{
				peer("10.0.0.2", 50052, RPCProtoMajor, 0, 5000),
				peer("10.0.0.3", 50052, RPCProtoMajor, 0, 3000),
			},
			wantRPCs: "10.0.0.2:50052,10.0.0.3:50052",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := SelectRPCServers(c.gpus, c.predict, c.peers, c.opts)
			if got.RPCServers != c.wantRPCs {
				t.Errorf("RPCServers = %q, want %q", got.RPCServers, c.wantRPCs)
			}
		})
	}
}

// TestSelectRPCServersMinUsefulMemory: a peer with less than minUsefulMemory
// free is never worth the hop, even as the only candidate.
func TestSelectRPCServersMinUsefulMemory(t *testing.T) {
	got := SelectRPCServers(
		[]ml.DeviceInfo{gpu(100)},
		mib(10000),
		[]Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 1000)},
		api.Options{},
	)
	if got.RPCServers != "" {
		t.Fatalf("RPCServers = %q, want empty (peer below minUsefulMemory)", got.RPCServers)
	}
}

// TestSelectRPCServersMinUsefulMemoryBoundary pins the floor at 1 GiB, not
// the earlier 2 GiB: a peer with 1200 MiB free -- below the old floor,
// above the current one -- must be usable.
func TestSelectRPCServersMinUsefulMemoryBoundary(t *testing.T) {
	got := SelectRPCServers(
		[]ml.DeviceInfo{gpu(100)},
		mib(10000),
		[]Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 1200)},
		api.Options{},
	)
	want := "10.0.0.2:50052"
	if got.RPCServers != want {
		t.Fatalf("RPCServers = %q, want %q (peer above the 1 GiB floor)", got.RPCServers, want)
	}
}

// TestSelectRPCServersPrefersLowLoad: two peers with equal free memory but
// different self-reported load. The idle one should be used to its full
// capacity and listed first; the busy one only covers the remainder.
func TestSelectRPCServersPrefersLowLoad(t *testing.T) {
	busy := peerAt("10.0.0.2", 5000, 0.9, 0)
	idle := peerAt("10.0.0.3", 5000, 0, 0)

	got := SelectRPCServers([]ml.DeviceInfo{gpu(100)}, mib(6000), []Peer{busy, idle}, api.Options{})
	want := "10.0.0.3:50052,10.0.0.2:50052"
	if got.RPCServers != want {
		t.Fatalf("RPCServers = %q, want %q", got.RPCServers, want)
	}
}

// TestSelectRPCServersPrefersLowLatency: two peers with equal free memory
// and load, but different measured link latency. The near one should be
// used to its full capacity and listed first.
func TestSelectRPCServersPrefersLowLatency(t *testing.T) {
	near := peerAt("10.0.0.2", 5000, 0, 0)
	far := peerAt("10.0.0.3", 5000, 0, 2*time.Second)

	got := SelectRPCServers([]ml.DeviceInfo{gpu(100)}, mib(6000), []Peer{near, far}, api.Options{})
	want := "10.0.0.2:50052,10.0.0.3:50052"
	if got.RPCServers != want {
		t.Fatalf("RPCServers = %q, want %q", got.RPCServers, want)
	}
}

// TestSelectRPCServersGreedyPolicy: same busy/idle equal-memory peers as
// TestSelectRPCServersPrefersLowLoad, but forced to "greedy" policy. Both
// peers are still needed to cover the shortfall, but greedy ignores load
// entirely -- ties on free memory break by input order, not by who's idle.
func TestSelectRPCServersGreedyPolicy(t *testing.T) {
	busy := peerAt("10.0.0.2", 5000, 0.9, 0)
	idle := peerAt("10.0.0.3", 5000, 0, 0)

	got := SelectRPCServers([]ml.DeviceInfo{gpu(100)}, mib(6000), []Peer{busy, idle},
		api.Options{Runner: api.Runner{RPCPlacement: "greedy"}})
	want := "10.0.0.2:50052,10.0.0.3:50052" // input order: busy first
	if got.RPCServers != want {
		t.Fatalf("RPCServers = %q, want %q", got.RPCServers, want)
	}
}

// TestSelectRPCServersGreedyStopsEarly: three peers (6000/5000/3000 MiB, all
// above minUsefulMemory) covering a 9457 MiB shortfall. Greedy stops once
// the two biggest cover it, leaving the third completely unused; water-fill
// (the default) spreads across all three instead. Same input, both
// policies exercised explicitly so the divergence is the point of the test,
// not incidental.
func TestSelectRPCServersGreedyStopsEarly(t *testing.T) {
	gpus := []ml.DeviceInfo{gpu(1000)} // 543 MiB local available, see TestSelectRPCServers
	peers := []Peer{
		peer("10.0.0.2", 50052, RPCProtoMajor, 0, 6000),
		peer("10.0.0.3", 50052, RPCProtoMajor, 0, 5000),
		peer("10.0.0.4", 50052, RPCProtoMajor, 0, 3000),
	}

	greedy := SelectRPCServers(gpus, mib(10000), peers, api.Options{Runner: api.Runner{RPCPlacement: "greedy"}})
	wantGreedy := "10.0.0.2:50052,10.0.0.3:50052" // .4 never needed, left idle
	if greedy.RPCServers != wantGreedy {
		t.Fatalf("greedy RPCServers = %q, want %q", greedy.RPCServers, wantGreedy)
	}

	waterfill := SelectRPCServers(gpus, mib(10000), peers, api.Options{Runner: api.Runner{RPCPlacement: "waterfill"}})
	wantWaterfill := "10.0.0.2:50052,10.0.0.3:50052,10.0.0.4:50052" // all three share the load
	if waterfill.RPCServers != wantWaterfill {
		t.Fatalf("waterfill RPCServers = %q, want %q", waterfill.RPCServers, wantWaterfill)
	}
}

// TestWaterFill exercises the bisection directly: equal rate/link splits
// proportionally to cap, and a shortfall no combination can fully cover
// still saturates every device at its own cap rather than returning zero.
func TestWaterFill(t *testing.T) {
	fracs := waterFill([]float64{1, 1}, []float64{0, 0}, []float64{0.6, 0.6})
	if math.Abs(fracs[0]-0.5) > 1e-6 || math.Abs(fracs[1]-0.5) > 1e-6 {
		t.Fatalf("equal devices under cap: got %v, want [0.5 0.5]", fracs)
	}

	fracs = waterFill([]float64{1, 1}, []float64{0, 0}, []float64{0.2, 0.3})
	if math.Abs(fracs[0]-0.2) > 1e-6 || math.Abs(fracs[1]-0.3) > 1e-6 {
		t.Fatalf("unsatisfiable: got %v, want [0.2 0.3] (both maxed at their cap)", fracs)
	}
}

// TestSelectReachableRPCServers: a peer that doesn't accept connections is
// dropped and the pick is redone without it, instead of being handed to
// llama-server (which aborts the whole load on it).
func TestSelectReachableRPCServers(t *testing.T) {
	down := map[string]bool{"10.0.0.2:50052": true}
	var mu sync.Mutex
	var dialed []string
	dial := func(addr string) error {
		mu.Lock()
		defer mu.Unlock()
		dialed = append(dialed, addr)
		if down[addr] {
			return errors.New("connect: no route to host")
		}
		return nil
	}
	gpus := []ml.DeviceInfo{gpu(100)}

	t.Run("unreachable peer is replaced by a reachable one", func(t *testing.T) {
		t.Setenv("OLLAMA_CLUSTER_PLACEMENT", "greedy")
		peers := []Peer{
			peer("10.0.0.2", 50052, RPCProtoMajor, 0, 20000), // biggest, greedy's first pick
			peer("10.0.0.3", 50052, RPCProtoMajor, 0, 9000),
		}
		got := SelectReachableRPCServers(gpus, mib(4000), peers, api.Options{}, dial)
		if got.RPCServers != "10.0.0.3:50052" {
			t.Errorf("RPCServers = %q, want 10.0.0.3:50052", got.RPCServers)
		}
	})

	t.Run("no reachable peer loads locally", func(t *testing.T) {
		peers := []Peer{peer("10.0.0.2", 50052, RPCProtoMajor, 0, 20000)}
		got := SelectReachableRPCServers(gpus, mib(4000), peers, api.Options{}, dial)
		if got.RPCServers != "" {
			t.Errorf("RPCServers = %q, want none", got.RPCServers)
		}
	})

	t.Run("caller's own RPCServers are not dialed", func(t *testing.T) {
		dialed = nil
		opts := api.Options{Runner: api.Runner{RPCServers: "10.0.0.2:50052"}}
		got := SelectReachableRPCServers(gpus, mib(4000), nil, opts, dial)
		if got.RPCServers != "10.0.0.2:50052" || len(dialed) != 0 {
			t.Errorf("RPCServers = %q, dialed %v; want passed through undialed", got.RPCServers, dialed)
		}
	})
}

// TestSelectRPCServersFillsLocalFirst: a 20 GiB GPU (17.7 GiB free) next to a
// big fast-enough peer and a small slow one. The default (waterfill) policy
// used to list both peers and let llama.cpp split by free memory, which put
// weights on the small peer; the fill keeps the GPU full, uses the big peer
// for the rest and leaves the small one out. Numbers are MiB:
// GPU room = 18125 - 457 reserve - 3072 margin = 14596; big peer room = 22528.
func TestSelectRPCServersFillsLocalFirst(t *testing.T) {
	gpus := []ml.DeviceInfo{gpu(18125)}
	big := peerAt("10.0.0.2", 25600, 0, time.Millisecond)
	small := peerAt("10.0.0.3", 12288, 0, 0) // idle and closer, so waterfill alone ranks it first
	peers := []Peer{small, big}

	for _, policy := range []string{"waterfill", "greedy"} {
		got := SelectRPCServers(gpus, mib(25600), peers, api.Options{Runner: api.Runner{RPCPlacement: policy}})
		if got.RPCServers != "10.0.0.2:50052" || got.TensorSplit != "11004,14596" {
			t.Fatalf("%s: got rpc %q split %q, want only the big peer, split 11004,14596", policy, got.RPCServers, got.TensorSplit)
		}
	}

	// Needs both peers: the big one first, then the small one for the rest.
	got := SelectRPCServers(gpus, mib(40960), peers, api.Options{})
	if got.RPCServers != "10.0.0.2:50052,10.0.0.3:50052" || got.TensorSplit != "22528,3836,14596" {
		t.Fatalf("both peers: got rpc %q split %q", got.RPCServers, got.TensorSplit)
	}

	// Does not fit anywhere: keep llama.cpp's split (and its --fit CPU spill).
	got = SelectRPCServers(gpus, mib(80000), peers, api.Options{})
	if got.TensorSplit != "" {
		t.Fatalf("model larger than all devices: split %q, want none", got.TensorSplit)
	}

	// A split the caller chose is never replaced.
	got = SelectRPCServers(gpus, mib(25600), peers, api.Options{Runner: api.Runner{TensorSplit: "1,1"}})
	if got.TensorSplit != "1,1" {
		t.Fatalf("explicit split replaced: %q", got.TensorSplit)
	}

	// Two local GPUs: their order is llama.cpp's, leave the split alone.
	got = SelectRPCServers([]ml.DeviceInfo{gpu(18125), gpu(18125)}, mib(60000), peers, api.Options{})
	if got.TensorSplit != "" {
		t.Fatalf("multi-GPU: split %q, want none", got.TensorSplit)
	}
}

// fakeRPCWorker listens on localhost and runs serve for each accepted
// connection; it returns the address to dial.
func fakeRPCWorker(t *testing.T, serve func(net.Conn)) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("no localhost listener: %v", err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				serve(c)
			}()
		}
	}()
	return ln.Addr().String()
}

// helloReply answers a HELLO the way rpc_serve_client does.
func helloReply(c net.Conn, major, minor byte) {
	req := make([]byte, 1+8+rpcConnCapsSize)
	if _, err := io.ReadFull(c, req); err != nil || req[0] != rpcCmdHello {
		return
	}
	rsp := make([]byte, 8+4+rpcConnCapsSize)
	binary.LittleEndian.PutUint64(rsp, 4+rpcConnCapsSize)
	rsp[8], rsp[9] = major, minor
	c.Write(rsp)
}

// TestDialRPC: a worker that accepts the TCP connection and then resets it
// (or hangs up, or speaks another protocol) must not pass the check, because
// llama-server aborts the whole load on it without naming the peer (#70).
func TestDialRPC(t *testing.T) {
	healthy := fakeRPCWorker(t, func(c net.Conn) { helloReply(c, RPCProtoMajor, RPCProtoMinor) })
	if err := DialRPC(healthy); err != nil {
		t.Errorf("healthy worker: %v", err)
	}

	resets := fakeRPCWorker(t, func(c net.Conn) { c.(*net.TCPConn).SetLinger(0) }) // close sends RST
	hangsUp := fakeRPCWorker(t, func(c net.Conn) {})
	oldMajor := fakeRPCWorker(t, func(c net.Conn) { helloReply(c, RPCProtoMajor-1, 0) })
	newerMinor := fakeRPCWorker(t, func(c net.Conn) { helloReply(c, RPCProtoMajor, RPCProtoMinor+1) })
	for name, addr := range map[string]string{"reset": resets, "hang up": hangsUp, "other major": oldMajor, "newer minor": newerMinor} {
		if err := DialRPC(addr); err == nil {
			t.Errorf("%s: DialRPC passed, want an error", name)
		}
	}
}

// TestPlacementLogsCandidates: one INFO line per decision that names every
// peer handed to placement, including the ones it could not use, so a peer
// that went missing from the pick can be explained afterwards (#71).
func TestPlacementLogsCandidates(t *testing.T) {
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(old) })

	now := time.Now()
	big := peerAt("10.0.0.2", 25600, 0, 0)
	big.LastSeen = now
	restarting := peerAt("10.0.0.3", 12288, 0, 0)
	restarting.RPCPort = 0
	old2 := peerAt("10.0.0.4", 12288, 0, 0)
	old2.ProtoMajor = RPCProtoMajor - 1

	got := SelectRPCServers([]ml.DeviceInfo{gpu(2000)}, mib(4000), []Peer{big, restarting, old2}, api.Options{})
	if got.RPCServers != "10.0.0.2:50052" {
		t.Fatalf("RPCServers = %q", got.RPCServers)
	}
	out := buf.String()
	if strings.Count(out, "cluster: placement") != 1 {
		t.Fatalf("want exactly one placement line, got:\n%s", out)
	}
	for _, want := range []string{"10.0.0.2 free=25.0 GiB rpc_port=50052", "beacon_age=0s: usable", "10.0.0.3 free=12.0 GiB rpc_port=0", "not sharing (rpc_port 0)", "speaks RPC protocol", "rpc_servers=10.0.0.2:50052"} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}

	buf.Reset()
	SelectRPCServers([]ml.DeviceInfo{gpu(8000)}, mib(4000), []Peer{big}, api.Options{})
	if buf.Len() != 0 {
		t.Errorf("logged although the model fits locally:\n%s", buf.String())
	}
}
