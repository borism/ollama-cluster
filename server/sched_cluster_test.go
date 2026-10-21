package server

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/format"
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
