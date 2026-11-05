package cluster

import (
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/borism/ollama-cluster/ml"
)

// weightsTableTag is the llama.cpp tag unsupportedWeights was read from.
// TestUnsupportedWeightsMatchesPin fails when LLAMA_CPP_VERSION moves, so a
// sync re-reads the functions each entry names and updates the table.
const weightsTableTag = "b11963"

// unsupportedWeight is one ggml tensor type a peer's backend refuses to
// compute. Over RPC the head never finds out: ggml-rpc.cpp's
// ggml_backend_rpc_device_supports_op returns true for everything (there is no
// RPC command to ask the peer), so llama.cpp places those layers on the peer
// anyway and the peer's ggml-rpc-server aborts at warm-up with "unsupported op"
// (#54). On a single machine the same check sends them to the CPU instead.
type unsupportedWeight struct {
	backend  string // ml.DeviceInfo.Library, as peers advertise it per device
	ggmlType string // GGML_TYPE_* without the prefix, e.g. "NVFP4"
	source   string // where llama.cpp refuses it, at weightsTableTag
}

// unsupportedWeights only lists types refused for MUL_MAT/MUL_MAT_ID, the ops
// a layer's weights go through on the device holding them. Refusals that
// depend on the device (BF16 on a Metal GPU without bfloat, Vulkan shared
// memory limits) or only affect GET_ROWS on the token embedding, which
// llama.cpp keeps on the head's CPU, are left out. Vulkan refuses no weight
// type for these ops at weightsTableTag. docs/cluster.mdx ("Model formats and
// backends") says which entries were seen failing on real hardware.
var unsupportedWeights = []unsupportedWeight{
	{"Metal", "NVFP4", "ggml_metal_supports_mul_mat_op in ggml-metal-device.m"},
	{"Metal", "TQ1_0", "ggml_metal_supports_mul_mat_op in ggml-metal-device.m"},
	{"CUDA", "TQ1_0", "MUL_MAT in ggml_backend_cuda_device_supports_op, ggml-cuda.cu"},
	{"CUDA", "TQ2_0", "MUL_MAT in ggml_backend_cuda_device_supports_op, ggml-cuda.cu"},
	// ggml-hip compiles ggml-cuda.cu, so ROCm shares CUDA's supports_op.
	{"ROCm", "TQ1_0", "MUL_MAT in ggml_backend_cuda_device_supports_op, ggml-cuda.cu"},
	{"ROCm", "TQ2_0", "MUL_MAT in ggml_backend_cuda_device_supports_op, ggml-cuda.cu"},
}

// ModelFormat is what placement checks against each peer's backend.
type ModelFormat struct {
	// Name is the model's GGUF file type, e.g. "NVFP4" or "Q4_K_M", for
	// messages.
	Name string
	// TensorTypes is every ggml type the model's tensors use, e.g. "F32",
	// "Q4_K", "NVFP4" (case does not matter).
	TensorTypes []string
}

// FormatIncompatibility returns "" if unsupportedWeights knows of no reason p
// can't hold layers of a model in format m, else why not. A device that does
// not report its backend (Library) is unknown and allowed, as before this
// check existed.
func (p Peer) FormatIncompatibility(m ModelFormat) string {
	var reasons []string
	for _, u := range unsupportedWeights {
		if !slices.ContainsFunc(p.Devices, func(d ml.DeviceInfo) bool { return strings.EqualFold(d.Library, u.backend) }) ||
			!slices.ContainsFunc(m.TensorTypes, func(t string) bool { return strings.EqualFold(t, u.ggmlType) }) {
			continue
		}
		reasons = append(reasons, fmt.Sprintf("its %s backend cannot run %s tensors (llama.cpp %s: %s)", u.backend, u.ggmlType, weightsTableTag, u.source))
	}
	return strings.Join(reasons, "; ")
}

// Backends is the distinct backends p's devices report, e.g. "Metal", or
// "unknown backend" when none does.
func (p Peer) Backends() string {
	var libs []string
	for _, d := range p.Devices {
		if d.Library != "" && !slices.Contains(libs, d.Library) {
			libs = append(libs, d.Library)
		}
	}
	if len(libs) == 0 {
		return "unknown backend"
	}
	return strings.Join(libs, "/")
}

// ExcludeIncompatiblePeers drops the peers whose backend cannot run m (see
// FormatIncompatibility), logging one line for each that names the format,
// the peer and why. When the model fits on gpus no peer is used, so it
// returns peers unchanged and logs nothing.
func ExcludeIncompatiblePeers(gpus []ml.DeviceInfo, predicted uint64, peers []Peer, m ModelFormat) []Peer {
	if localAvailable(gpus) >= predicted {
		return peers
	}
	return slices.DeleteFunc(slices.Clone(peers), func(p Peer) bool {
		reason := p.FormatIncompatibility(m)
		if reason != "" {
			slog.Info("cluster: not placing layers on a peer that cannot run this model's format",
				"format", m.Name, "peer", rpcAddr(p), "id", p.ID, "reason", reason)
		}
		return reason != ""
	})
}
