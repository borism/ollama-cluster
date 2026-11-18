package cluster

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/borism/ollama-cluster/ml"
)

func peerWithBackend(addr, library string) Peer {
	p := peerAt(addr, 16384, 0, 0)
	p.ID = "id-" + addr
	p.Devices[0].Library = library
	return p
}

var (
	nvfp4Model = ModelFormat{Name: "NVFP4", TensorTypes: []string{"F32", "Q8_0", "NVFP4"}}
	q4kModel   = ModelFormat{Name: "Q4_K_M", TensorTypes: []string{"F32", "Q4_K", "Q6_K"}}
)

func TestFormatIncompatibility(t *testing.T) {
	cases := []struct {
		backend string
		model   ModelFormat
		refused bool
	}{
		{"Metal", nvfp4Model, true},
		{"Metal", q4kModel, false},
		{"Metal", ModelFormat{Name: "TQ1_0", TensorTypes: []string{"tq1_0"}}, true},
		{"Metal", ModelFormat{Name: "MXFP4_MOE", TensorTypes: []string{"MXFP4", "IQ4_NL"}}, false},
		{"CUDA", nvfp4Model, false},
		{"CUDA", ModelFormat{Name: "TQ2_0", TensorTypes: []string{"TQ2_0"}}, true},
		{"ROCm", ModelFormat{Name: "TQ2_0", TensorTypes: []string{"TQ2_0"}}, true},
		{"Vulkan", nvfp4Model, false},
		{"Vulkan", ModelFormat{Name: "TQ1_0", TensorTypes: []string{"TQ1_0"}}, false},
		// A peer whose devices don't report a backend is unknown: allowed.
		{"", nvfp4Model, false},
	}
	for _, c := range cases {
		reason := peerWithBackend("10.0.0.2", c.backend).FormatIncompatibility(c.model)
		if (reason != "") != c.refused {
			t.Errorf("%q backend, %s model: reason %q, want refused=%v", c.backend, c.model.Name, reason, c.refused)
		}
	}

	reason := peerWithBackend("10.0.0.2", "Metal").FormatIncompatibility(nvfp4Model)
	for _, want := range []string{"Metal", "NVFP4", weightsTableTag, "ggml_metal_supports_mul_mat_op"} {
		if !strings.Contains(reason, want) {
			t.Errorf("reason %q lacks %q", reason, want)
		}
	}

	// One refusing device is enough: llama.cpp spreads layers over all of a
	// peer's devices.
	mixed := peerWithBackend("10.0.0.3", "Vulkan")
	mixed.Devices = append(mixed.Devices, ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}, FreeMemory: 1 << 30})
	if mixed.FormatIncompatibility(nvfp4Model) == "" {
		t.Error("peer with a Metal device kept for an NVFP4 model")
	}
}

func TestExcludeIncompatiblePeers(t *testing.T) {
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(old) })

	metal := peerWithBackend("10.0.0.2", "Metal")
	unknown := peerWithBackend("10.0.0.3", "")
	cuda := peerWithBackend("10.0.0.4", "CUDA")
	peers := []Peer{metal, unknown, cuda}
	small := []ml.DeviceInfo{gpu(2000)}

	got := ExcludeIncompatiblePeers(small, mib(20000), peers, nvfp4Model)
	if len(got) != 2 || got[0].Addr != "10.0.0.3" || got[1].Addr != "10.0.0.4" {
		t.Fatalf("NVFP4: kept %+v, want the unknown-backend and CUDA peers", got)
	}
	if len(peers) != 3 {
		t.Fatal("caller's slice was modified")
	}
	out := buf.String()
	if strings.Count(out, "cannot run this model's format") != 1 {
		t.Fatalf("want one log line, got:\n%s", out)
	}
	for _, want := range []string{"format=NVFP4", "peer=10.0.0.2:50052", "its Metal backend cannot run NVFP4 tensors"} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}

	buf.Reset()
	if got := ExcludeIncompatiblePeers(small, mib(20000), peers, q4kModel); len(got) != 3 {
		t.Errorf("Q4_K: kept %d peers, want all 3", len(got))
	}
	if buf.Len() != 0 {
		t.Errorf("Q4_K logged:\n%s", buf.String())
	}

	// Fits locally: no peer is used, nothing to say.
	if got := ExcludeIncompatiblePeers([]ml.DeviceInfo{gpu(30000)}, mib(20000), peers, nvfp4Model); len(got) != 3 {
		t.Errorf("fits locally: kept %d peers, want all 3", len(got))
	}
	if buf.Len() != 0 {
		t.Errorf("logged although the model fits locally:\n%s", buf.String())
	}
}

// Peers already advertise each device's backend: ml.DeviceID.Library goes
// out as "backend" in the beacon. A beacon without it (an older or unusual
// build) leaves the backend unknown, which placement allows.
func TestAnnouncementCarriesBackend(t *testing.T) {
	buf, err := json.Marshal(announcement{ID: "p", Devices: []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}, FreeMemory: 8 << 30}}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(buf), `"backend":"Metal"`) {
		t.Fatalf("announcement lacks the device backend: %s", buf)
	}
	var a announcement
	if err := json.Unmarshal(buf, &a); err != nil {
		t.Fatal(err)
	}
	if got := (Peer{Devices: a.Devices}).Backends(); got != "Metal" {
		t.Fatalf("Backends() = %q", got)
	}

	var old announcement
	if err := json.Unmarshal([]byte(`{"id":"q","rpc_port":50052,"devices":[{"id":"0","name":"GPU0","total_memory":8589934592,"free_memory":8589934592}],"proto_major":8}`), &old); err != nil {
		t.Fatal(err)
	}
	p := Peer{Devices: old.Devices}
	if got := p.Backends(); got != "unknown backend" {
		t.Errorf("Backends() = %q", got)
	}
	if reason := p.FormatIncompatibility(nvfp4Model); reason != "" {
		t.Errorf("unknown backend refused: %q", reason)
	}
}

// A sync PR that bumps LLAMA_CPP_VERSION must re-read the functions named in
// unsupportedWeights at the new tag (and the Vulkan supports_op, which
// refuses no weight type at b11963), update the table and docs/cluster.mdx
// ("Model formats and backends"), then weightsTableTag.
func TestUnsupportedWeightsMatchesPin(t *testing.T) {
	b, err := os.ReadFile("../LLAMA_CPP_VERSION")
	if err != nil {
		t.Fatal(err)
	}
	if pin := strings.TrimSpace(string(b)); pin != weightsTableTag {
		t.Fatalf("unsupportedWeights (cluster/capability.go) was read from llama.cpp %s but LLAMA_CPP_VERSION is %s; re-check it at %s and update weightsTableTag", weightsTableTag, pin, pin)
	}
}
