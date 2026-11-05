package cluster

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/ml"
)

// Numbers are MiB. gpu(20000) leaves 19543 after the 457 reserve, so a
// 20000 model is 457 short (0.023 of it) and a 24000 one 4457 (0.186).
var (
	minSpillGPUs  = []ml.DeviceInfo{gpu(20000)}
	minSpillPeers = []Peer{peerAt("10.0.0.2", 25600, 0, 0)}
)

// captureLog sends slog output to a buffer for the rest of the test.
func captureLogAt(t *testing.T, level slog.Level) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: level})))
	t.Cleanup(func() { slog.SetDefault(old) })
	return &buf
}

func TestMinSpillOffByDefault(t *testing.T) {
	t.Setenv("OLLAMA_CLUSTER_MIN_SPILL", "")
	buf := captureLogAt(t, slog.LevelDebug)

	got := SelectRPCServers(minSpillGPUs, mib(20000), minSpillPeers, api.Options{})
	if got.RPCServers != "10.0.0.2:50052" {
		t.Fatalf("RPCServers = %q, want the peer: with no threshold even a 457 MiB shortfall spills", got.RPCServers)
	}
	if strings.Contains(buf.String(), "MIN_SPILL") {
		t.Errorf("logged a min-spill decision while it is off:\n%s", buf)
	}
}

func TestMinSpillKeepsSmallShortfallLocal(t *testing.T) {
	t.Setenv("OLLAMA_CLUSTER_MIN_SPILL", "0.05")
	t.Setenv("OLLAMA_CLUSTER_PLACEMENT", "greedy") // a server-wide policy is not a per-request ask
	buf := captureLogAt(t, slog.LevelInfo)

	got := SelectReachableRPCServers(minSpillGPUs, mib(20000), minSpillPeers, api.Options{}, func(addr string) error {
		t.Errorf("dialed %s although the model stays local", addr)
		return nil
	})
	if got.RPCServers != "" || got.TensorSplit != "" {
		t.Fatalf("got rpc %q split %q, want a local-only load", got.RPCServers, got.TensorSplit)
	}

	out := buf.String()
	if n := strings.Count(out, "\n"); n != 1 {
		t.Fatalf("want one log line, got %d:\n%s", n, out)
	}
	for _, want := range []string{"loading local-only", `predicted="19.5 GiB"`, `local="19.1 GiB"`, `shortfall="457.0 MiB"`, "cpu_share=0.023", "min_spill=0.05"} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
}

func TestMinSpillSpillsLargeShortfall(t *testing.T) {
	t.Setenv("OLLAMA_CLUSTER_MIN_SPILL", "0.05")
	buf := captureLogAt(t, slog.LevelInfo)

	got := SelectRPCServers(minSpillGPUs, mib(24000), minSpillPeers, api.Options{})
	if got.RPCServers != "10.0.0.2:50052" {
		t.Fatalf("RPCServers = %q, want the peer for a 0.186 shortfall", got.RPCServers)
	}
	if out := buf.String(); strings.Contains(out, "MIN_SPILL") || !strings.Contains(out, "cluster: placement") {
		t.Errorf("want only the usual placement line:\n%s", out)
	}
}

// TestMinSpillBoundary: a shortfall exactly at the threshold stays local, one
// just over it spills. gpu(19457) leaves 19000 MiB; a 20000 MiB model is then
// exactly 0.05 short.
func TestMinSpillBoundary(t *testing.T) {
	t.Setenv("OLLAMA_CLUSTER_MIN_SPILL", "0.05")
	gpus := []ml.DeviceInfo{gpu(19457)}

	if got := SelectRPCServers(gpus, mib(20000), minSpillPeers, api.Options{}); got.RPCServers != "" {
		t.Errorf("at the threshold: RPCServers = %q, want local-only", got.RPCServers)
	}
	if got := SelectRPCServers(gpus, mib(20001), minSpillPeers, api.Options{}); got.RPCServers == "" {
		t.Error("just over the threshold: want a peer")
	}
}

// TestMinSpillRespectsExplicitChoices: the threshold never overrides what a
// request chose itself.
func TestMinSpillRespectsExplicitChoices(t *testing.T) {
	t.Setenv("OLLAMA_CLUSTER_MIN_SPILL", "1") // would keep any model local
	yes := true

	t.Run("rpc_servers is not overridden", func(t *testing.T) {
		opts := api.Options{Runner: api.Runner{RPCServers: "192.0.2.9:50052", TensorSplit: "1,3"}}
		for name, got := range map[string]api.Options{
			"select": SelectRPCServers(minSpillGPUs, mib(20000), minSpillPeers, opts),
			"reachable": SelectReachableRPCServers(minSpillGPUs, mib(20000), minSpillPeers, opts, func(addr string) error {
				t.Errorf("dialed %s, want the caller's rpc_servers passed through", addr)
				return nil
			}),
		} {
			if got.RPCServers != opts.RPCServers || got.TensorSplit != opts.TensorSplit {
				t.Errorf("%s: got rpc %q split %q, want the request's own", name, got.RPCServers, got.TensorSplit)
			}
		}
	})

	for name, opts := range map[string]api.Options{
		"rpc_auto true": {Runner: api.Runner{RPCAuto: &yes}},
		"rpc_placement": {Runner: api.Runner{RPCPlacement: "greedy"}},
	} {
		t.Run(name+" still spills", func(t *testing.T) {
			buf := captureLogAt(t, slog.LevelDebug)
			if got := SelectRPCServers(minSpillGPUs, mib(20000), minSpillPeers, opts); got.RPCServers != "10.0.0.2:50052" {
				t.Fatalf("RPCServers = %q, want the peer the request asked for", got.RPCServers)
			}
			if !strings.Contains(buf.String(), "the request asked for the cluster") {
				t.Errorf("want a debug line saying why the threshold was not applied:\n%s", buf)
			}
		})
	}
}

func TestLocalCPUShare(t *testing.T) {
	for _, c := range []struct {
		local, predicted, shortfall uint64
		share                       float64
	}{
		{mib(19000), mib(20000), mib(1000), 0.05},
		{mib(20000), mib(20000), 0, 0},
		{mib(30000), mib(20000), 0, 0},
		{0, mib(20000), mib(20000), 1}, // no GPU: all of it on the CPU
		{0, 0, 0, 0},
	} {
		if shortfall, share := localCPUShare(c.local, c.predicted); shortfall != c.shortfall || share != c.share {
			t.Errorf("localCPUShare(%d, %d) = %d, %v; want %d, %v", c.local, c.predicted, shortfall, share, c.shortfall, c.share)
		}
	}
}
