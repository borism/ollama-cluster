package server

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/cluster"
	"github.com/borism/ollama-cluster/format"
	"github.com/borism/ollama-cluster/fs/gguf"
	"github.com/borism/ollama-cluster/llm"
	"github.com/borism/ollama-cluster/ml"
)

func TestClusterUsageFromRPC(t *testing.T) {
	cases := []struct {
		name    string
		servers string
		vram    map[string]uint64
		want    []clusterPeerUsage
	}{
		{name: "no rpc servers", servers: "", vram: map[string]uint64{"RPC0": 100}, want: nil},
		{name: "no rpc usage yet", servers: "10.0.0.5:50052", vram: nil, want: nil},
		{
			name:    "index maps to RPCn in list order",
			servers: "10.0.0.5:50052,10.0.0.6:50052",
			vram:    map[string]uint64{"RPC0": 100, "RPC1": 200},
			want: []clusterPeerUsage{
				{addr: "10.0.0.5:50052", size: 100},
				{addr: "10.0.0.6:50052", size: 200},
			},
		},
		{
			name:    "a peer with zero bytes used is omitted",
			servers: "10.0.0.5:50052,10.0.0.6:50052",
			vram:    map[string]uint64{"RPC0": 100},
			want:    []clusterPeerUsage{{addr: "10.0.0.5:50052", size: 100}},
		},
		{
			name:    "whitespace around addresses is trimmed",
			servers: " 10.0.0.5:50052 , 10.0.0.6:50052 ",
			vram:    map[string]uint64{"RPC1": 200},
			want:    []clusterPeerUsage{{addr: "10.0.0.6:50052", size: 200}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, clusterUsageFromRPC(tc.servers, tc.vram))
		})
	}
}

// TestLoadedModelsClusterPeers is an end-to-end check that loadedModels()
// actually wires clusterUsageFromRPC to the runner's own --rpc list and
// llm.LlamaServer.RPCVRAM, not just that the pure function above is correct.
// The request options carry no RPCServers, as with automatic spillover:
// cluster.SelectRPCServers only fills in the launch options.
func TestLoadedModelsClusterPeers(t *testing.T) {
	s := InitScheduler(t.Context())

	runner := &runnerRef{
		modelPath: "/fake/spilled/model",
		llama: &mockLlm{
			totalSize:  10 * format.GigaByte,
			vramSize:   10 * format.GigaByte,
			rpcVRAM:    map[string]uint64{"RPC0": 6 * format.GigaByte},
			rpcServers: "10.0.0.5:50052",
		},
		Options:         &api.Options{},
		sessionDuration: 10 * time.Millisecond,
		expiresAt:       time.Now().Add(time.Minute),
	}
	runner.model.Store(&Model{Name: "spilled", ModelPath: "/fake/spilled/model"})

	s.loadedMu.Lock()
	s.loaded["/fake/spilled/model"] = runner
	s.loadedMu.Unlock()

	models := s.loadedModels()
	require.Len(t, models, 1)
	require.Equal(t, []clusterPeerUsage{{addr: "10.0.0.5:50052", size: int64(6 * format.GigaByte)}}, models[0].clusterPeers)
}

// TestSchedLoadRPCPeerDropped: when a load fails because llama-server lost an
// --rpc peer (a worker that exited mid-load restarts in about a second, #82),
// load waits for the peer to answer the RPC handshake again and retries once
// with the same peers. A peer that stays down goes to the existing retry
// without it, and there is never a second same-peer retry.
func TestSchedLoadRPCPeerDropped(t *testing.T) {
	const peer = "10.0.0.5:50052"
	crashed := errors.New("llama-server process has terminated: signal: aborted: Remote RPC server crashed or returned malformed response")
	refused := errors.New("llama-server process has terminated: exit status 1: Failed to connect to " + peer)

	// Whether the peer refuses a handshake, given how many loads have failed
	// so far and how many handshakes it has had since the last one.
	backAfterOne := func(failed, dials int) bool { return failed > 0 && dials < 1 }
	staysDown := func(failed, _ int) bool { return failed > 0 }
	alwaysUp := func(int, int) bool { return false }

	cases := []struct {
		name string
		// rpcServers is the request's own rpc_servers ("": the scheduler picks).
		rpcServers string
		// fails is each attempt's load error (nil, or past the end: it loads).
		fails    []error
		peerDown func(failed, dials int) bool
		// beaconDown swaps in a beacon showing the peer not sharing (rpc_port 0,
		// as one sent while its worker restarts would) after the first launch.
		beaconDown   bool
		wantLaunched []string
		wantErr      error
		wantLog      []string
	}{
		{
			name:         "peer back mid-load: one retry with the same peers",
			fails:        []error{crashed},
			peerDown:     backAfterOne,
			beaconDown:   true,
			wantLaunched: []string{peer, peer},
			wantLog:      []string{"rpc peer dropped during load, waiting for it to come back", "rpc peers are back, retrying the load once with the same peers"},
		},
		{
			name:         "peer down at connect, then back: one retry with the same peers",
			fails:        []error{refused},
			peerDown:     backAfterOne,
			wantLaunched: []string{peer, peer},
			wantLog:      []string{"rpc peers are back"},
		},
		{
			name:         "peer stays down: retried without it",
			fails:        []error{crashed},
			peerDown:     staysDown,
			wantLaunched: []string{peer, ""},
			wantLog:      []string{"rpc peer did not come back", "down=" + peer, "retrying without it", "addr=" + peer},
		},
		{
			// The crash recurs with the peer answering: no second same-peer
			// retry, but the crashed-peer retry without it (#55) still runs.
			name:         "drops again after its retry: no second same-peer retry",
			fails:        []error{crashed, crashed, crashed},
			peerDown:     alwaysUp,
			wantLaunched: []string{peer, peer, ""},
			wantErr:      crashed,
			wantLog:      []string{"not retrying with the same peers a second time", "peer crashed during load, retrying without it"},
		},
		{
			name:         "drops again and stays down: retried without it",
			fails:        []error{crashed, crashed},
			peerDown:     func(failed, dials int) bool { return failed == 1 && dials < 1 || failed > 1 },
			wantLaunched: []string{peer, peer, ""},
			wantLog:      []string{"not retrying with the same peers a second time", "retrying without it"},
		},
		{
			name:         "request's own rpc_servers: same-peer retry",
			rpcServers:   peer,
			fails:        []error{crashed},
			peerDown:     backAfterOne,
			wantLaunched: []string{peer, peer},
		},
		{
			name:         "request's own rpc_servers, peer stays down: not left out",
			rpcServers:   peer,
			fails:        []error{crashed},
			peerDown:     staysDown,
			wantLaunched: []string{peer},
			wantErr:      crashed,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logs := &lockedBuffer{}
			old := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(logs, nil)))
			defer slog.SetDefault(old)

			ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
			defer done()
			s := InitScheduler(ctx)
			s.rpcPeerWait = time.Second
			p := cluster.Peer{
				ID: "p1", Addr: "10.0.0.5", RPCPort: 50052,
				ProtoMajor: cluster.RPCProtoMajor, ProtoMinor: cluster.RPCProtoMinor,
				Devices: []ml.DeviceInfo{{TotalMemory: 64 * format.GigaByte, FreeMemory: 64 * format.GigaByte}},
			}
			s.clusterTable.Store(cluster.NewStaticTable(p))

			var failed, dials atomic.Int32
			s.dialRPCFn = func(addr string) error {
				if addr != peer {
					t.Errorf("dialed %q", addr)
				}
				if tc.peerDown(int(failed.Load()), int(dials.Add(1)-1)) {
					return errors.New("connection refused")
				}
				return nil
			}

			var launched, splits []string
			s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *gguf.Model, _ []string, _ []string, opts api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
				launched = append(launched, opts.RPCServers)
				splits = append(splits, opts.TensorSplit)
				m := &mockLlm{modelPath: model}
				if n := len(launched); n <= len(tc.fails) && tc.fails[n-1] != nil {
					m.loadErr = tc.fails[n-1]
					failed.Add(1)
					dials.Store(0)
					if n == 1 && tc.beaconDown {
						p.RPCPort = 0
						s.clusterTable.Store(cluster.NewStaticTable(p))
					}
				}
				return m, nil
			}

			scenario := newScenarioRequest(t, ctx, "m", 1*format.GigaByte, nil, nil)
			scenario.req.opts.RPCServers = tc.rpcServers
			gpus := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}}}
			s.load(scenario.req, getSystemInfoFn(), gpus, false)

			require.Equal(t, tc.wantLaunched, launched)
			if len(launched) > 1 && launched[1] == launched[0] {
				require.Equal(t, splits[0], splits[1], "the same-peer retry keeps the tensor split")
			}
			select {
			case err := <-scenario.req.errCh:
				require.ErrorIs(t, err, tc.wantErr)
			case <-scenario.req.successCh:
				require.NoError(t, tc.wantErr, "load succeeded")
			case <-time.After(time.Second):
				t.Fatal("load neither failed nor succeeded")
			}
			out := logs.String()
			for _, want := range tc.wantLog {
				require.Contains(t, out, want, out)
			}
			require.LessOrEqual(t, strings.Count(out, "waiting for it to come back"), 1, out)
		})
	}
}

// TestSchedLoadLocalOnlyFailureNotRetried: a load without --rpc peers is not
// waited on or retried, whatever its error says.
func TestSchedLoadLocalOnlyFailureNotRetried(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.dialRPCFn = func(addr string) error {
		t.Errorf("dialed %q for a local-only load", addr)
		return nil
	}
	loadErr := errors.New("llama-server process has terminated: signal: aborted: Remote RPC server crashed or returned malformed response")
	var launches int
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *gguf.Model, _ []string, _ []string, opts api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		launches++
		require.Empty(t, opts.RPCServers)
		return &mockLlm{modelPath: model, loadErr: loadErr}, nil
	}

	scenario := newScenarioRequest(t, ctx, "m", 1*format.GigaByte, nil, nil)
	start := time.Now()
	s.load(scenario.req, getSystemInfoFn(), []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}}}, false)

	require.Equal(t, 1, launches)
	require.Less(t, time.Since(start), s.rpcPeerWait)
	select {
	case err := <-scenario.req.errCh:
		require.ErrorIs(t, err, loadErr)
	case <-time.After(time.Second):
		t.Fatal("expected the load error")
	}
}

// TestAwaitRPCPeers: polls the peers that have not answered until they do or
// the wait is over, dials once with no wait, and stops when ctx is done.
func TestAwaitRPCPeers(t *testing.T) {
	const up, down = "10.0.0.5:50052", "10.0.0.6:50052"
	var upDials, downDials atomic.Int32
	s := &Scheduler{dialRPCFn: func(addr string) error {
		if addr == up {
			upDials.Add(1)
			return nil
		}
		downDials.Add(1)
		return errors.New("connection refused")
	}}
	addrs := []string{up, down}

	require.Equal(t, []string{down}, s.awaitRPCPeers(t.Context(), addrs, 0))
	require.Equal(t, int32(1), upDials.Load())
	require.Equal(t, int32(1), downDials.Load())

	upDials.Store(0)
	downDials.Store(0)
	start := time.Now()
	require.Equal(t, []string{down}, s.awaitRPCPeers(t.Context(), addrs, 300*time.Millisecond))
	require.GreaterOrEqual(t, time.Since(start), 300*time.Millisecond)
	require.Less(t, time.Since(start), 2*time.Second)
	require.Equal(t, int32(1), upDials.Load(), "a peer that answered is not dialed again")
	require.GreaterOrEqual(t, downDials.Load(), int32(2), "a peer that is down is polled")

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	start = time.Now()
	require.Equal(t, []string{down}, s.awaitRPCPeers(ctx, addrs, time.Minute))
	require.Less(t, time.Since(start), time.Second, "a done context ends the wait")
}
