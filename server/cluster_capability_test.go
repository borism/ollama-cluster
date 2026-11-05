package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/cluster"
	"github.com/borism/ollama-cluster/format"
	"github.com/borism/ollama-cluster/fs/gguf"
	gguftest "github.com/borism/ollama-cluster/internal/testutil/gguf"
	"github.com/borism/ollama-cluster/llm"
	"github.com/borism/ollama-cluster/ml"
)

// rpcCrashErr is how a head's load fails when a peer's ggml-rpc-server aborts
// at warm-up (#54): the peer logs "unsupported op", the head only this.
var rpcCrashErr = errors.New("llama-server process has terminated: signal: aborted: " +
	"/src/ggml/src/ggml-rpc/ggml-rpc.cpp:685: Remote RPC server crashed or returned malformed response")

func TestCrashedRPCPeers(t *testing.T) {
	if got := crashedRPCPeers(rpcCrashErr.Error(), "10.0.0.2:50052"); len(got) != 1 || got[0] != "10.0.0.2:50052" {
		t.Errorf("one peer: got %v", got)
	}
	// The abort doesn't name the peer, so with several every one is a suspect.
	if got := crashedRPCPeers(rpcCrashErr.Error(), "10.0.0.2:50052,10.0.0.3:50052"); len(got) != 2 {
		t.Errorf("two peers: got %v", got)
	}
	if got := crashedRPCPeers(rpcCrashErr.Error(), ""); got != nil {
		t.Errorf("no peers: got %v", got)
	}
	if got := crashedRPCPeers("cudaMalloc failed: out of memory", "10.0.0.2:50052"); got != nil {
		t.Errorf("unrelated error: got %v", got)
	}
}

func TestExplainRPCCrash(t *testing.T) {
	nvfp4 := cluster.ModelFormat{Name: "NVFP4", TensorTypes: []string{"F32", "NVFP4"}}
	peers := []cluster.Peer{
		{Addr: "10.0.0.2", RPCPort: 50052, Devices: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}}}},
		{Addr: "10.0.0.3", RPCPort: 50052, Devices: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Vulkan"}}}},
	}

	err := explainRPCCrash(rpcCrashErr, []string{"10.0.0.2:50052"}, peers, nvfp4)
	require.ErrorIs(t, err, rpcCrashErr)
	for _, want := range []string{"cluster peer 10.0.0.2:50052 (Metal)", "this NVFP4 model", "its Metal backend cannot run NVFP4 tensors"} {
		require.Contains(t, err.Error(), want)
	}

	// No table entry explains it: say what the head knows and where to look.
	err = explainRPCCrash(rpcCrashErr, []string{"10.0.0.3:50052"}, peers, nvfp4)
	require.Contains(t, err.Error(), "cluster peer 10.0.0.3:50052 (Vulkan) failed while loading this NVFP4 model")
	require.Contains(t, err.Error(), `"unsupported op" in that peer's server log`)

	err = explainRPCCrash(rpcCrashErr, []string{"10.0.0.9:50052"}, peers, nvfp4)
	require.Contains(t, err.Error(), "10.0.0.9:50052 (unknown backend)")

	err = explainRPCCrash(rpcCrashErr, []string{"10.0.0.2:50052", "10.0.0.3:50052"}, peers, nvfp4)
	require.ErrorIs(t, err, rpcCrashErr)
	for _, want := range []string{"10.0.0.2:50052 (Metal), 10.0.0.3:50052 (Vulkan)", "llama.cpp does not say which", "10.0.0.2:50052: its Metal backend cannot run NVFP4"} {
		require.Contains(t, err.Error(), want)
	}
}

// helloRPCWorker answers cluster.DialRPC's HELLO so a peer passes the
// reachability check; the mock llama-server decides how the load goes.
func helloRPCWorker(t *testing.T) (addr string, port int) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			req := make([]byte, 1+8+24)
			if _, err := io.ReadFull(c, req); err == nil {
				rsp := make([]byte, 8+4+24)
				binary.LittleEndian.PutUint64(rsp, 4+24)
				rsp[8], rsp[9] = cluster.RPCProtoMajor, cluster.RPCProtoMinor
				c.Write(rsp)
			}
			c.Close()
		}
	}()
	port = ln.Addr().(*net.TCPAddr).Port
	return fmt.Sprintf("127.0.0.1:%d", port), port
}

// nvfp4ScenarioRequest is newScenarioRequest with a model whose layer weights
// are NVFP4, like flint:40b (#54).
func nvfp4ScenarioRequest(t *testing.T, ctx context.Context) *reqBundle {
	t.Helper()
	b := newScenarioRequest(t, ctx, "flint", 1*format.GigaByte, nil, nil)
	nvfp4 := gguf.TensorTypeNVFP4
	weights := gguftest.Tensor{Name: "blk.0.attn.weight", Type: nvfp4, Shape: []uint64{64, 1, 1, 1}}
	p, _ := createBinFile(t, gguftest.KV{
		"general.architecture":          "llama",
		"general.file_type":             uint32(39), // LLAMA_FTYPE_MOSTLY_NVFP4
		"llama.context_length":          uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*gguftest.Tensor{
		{Name: weights.Name, Type: nvfp4, Shape: weights.Shape, WriterTo: bytes.NewReader(make([]byte, weights.Size()))},
		{Name: "output.weight", Type: gguf.TensorTypeF32, Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})
	b.req.model.ModelPath = p
	return b
}

func clusterPeerWith(port int, library string) cluster.Peer {
	return cluster.Peer{
		ID: "p-" + library, Addr: "127.0.0.1", RPCPort: port,
		ProtoMajor: cluster.RPCProtoMajor, ProtoMinor: cluster.RPCProtoMinor,
		Devices: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: library}, TotalMemory: 64 * format.GigaByte, FreeMemory: 64 * format.GigaByte}},
	}
}

func TestClusterModelFormat(t *testing.T) {
	b := nvfp4ScenarioRequest(t, t.Context())
	f, err := llm.LoadModel(b.req.model.ModelPath, 1024)
	require.NoError(t, err)
	got := clusterModelFormat(f)
	require.Equal(t, "NVFP4", got.Name)
	require.ElementsMatch(t, []string{"NVFP4", "F32"}, got.TensorTypes)
	require.Equal(t, "unknown", clusterModelFormat(nil).Name)
}

// A Metal peer is never handed an NVFP4 model's layers; a peer whose backend
// is unknown still is.
func TestSchedLoadSkipsPeerThatCannotRunFormat(t *testing.T) {
	for _, tc := range []struct {
		library string
		used    bool
	}{{"Metal", false}, {"", true}} {
		addr, port := helloRPCWorker(t)
		ctx, done := context.WithTimeout(t.Context(), time.Second)
		s := InitScheduler(ctx)
		s.clusterTable.Store(cluster.NewStaticTable(clusterPeerWith(port, tc.library)))
		var launched []string
		s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *gguf.Model, _ []string, _ []string, opts api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
			launched = append(launched, opts.RPCServers)
			return &mockLlm{modelPath: model}, nil
		}
		scenario := nvfp4ScenarioRequest(t, ctx)
		s.load(scenario.req, getSystemInfoFn(), []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}}, false)

		want := ""
		if tc.used {
			want = addr
		}
		require.Equal(t, []string{want}, launched, "peer backend %q", tc.library)
		done()
	}
}

// A peer whose ggml-rpc-server aborts mid-load (here a backend the table
// doesn't know about) gets one retry without it; with rpc_servers set by the
// request there is no retry, and the error names format, peer and reason.
func TestSchedLoadRetriesWithoutCrashedRPCPeer(t *testing.T) {
	addr, port := helloRPCWorker(t)
	for _, explicit := range []bool{false, true} {
		ctx, done := context.WithTimeout(t.Context(), time.Second)
		s := InitScheduler(ctx)
		s.clusterTable.Store(cluster.NewStaticTable(clusterPeerWith(port, "Vulkan")))
		var launched []string
		s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *gguf.Model, _ []string, _ []string, opts api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
			launched = append(launched, opts.RPCServers)
			m := &mockLlm{modelPath: model}
			if opts.RPCServers != "" {
				m.loadErr = rpcCrashErr
			}
			return m, nil
		}
		scenario := nvfp4ScenarioRequest(t, ctx)
		if explicit {
			scenario.req.opts.RPCServers = addr
		}
		s.load(scenario.req, getSystemInfoFn(), []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}}, false)

		if explicit {
			require.Equal(t, []string{addr}, launched)
			select {
			case err := <-scenario.req.errCh:
				require.ErrorIs(t, err, rpcCrashErr)
				msg := err.Error()
				for _, want := range []string{"cluster peer " + addr + " (Vulkan)", "this NVFP4 model", "unsupported op"} {
					if !strings.Contains(msg, want) {
						t.Errorf("error %q lacks %q", msg, want)
					}
				}
			case <-time.After(100 * time.Millisecond):
				t.Fatal("expected an error")
			}
		} else {
			require.Equal(t, []string{addr, ""}, launched)
			select {
			case <-scenario.req.successCh:
			case err := <-scenario.req.errCh:
				t.Fatalf("unexpected error %v", err)
			case <-time.After(time.Second):
				t.Fatal("expected successful load")
			}
		}
		done()
	}
}
