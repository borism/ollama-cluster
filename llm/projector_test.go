package llm

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/fs/gguf"
	gguftest "github.com/borism/ollama-cluster/internal/testutil/gguf"
	"github.com/borism/ollama-cluster/ml"
)

func projectorOpt(v bool) *bool { return &v }

func TestProjectorEnabled(t *testing.T) {
	for _, c := range []struct {
		name string
		env  string
		opt  *bool
		want bool
	}{
		{"default loads it", "", nil, true},
		{"env turns it off", "1", nil, false},
		{"option off", "", projectorOpt(false), false},
		{"option on beats env", "1", projectorOpt(true), true},
		{"option off beats env off", "1", projectorOpt(false), false},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("OLLAMA_NO_PROJECTOR", c.env)
			if got := ProjectorEnabled(api.Options{Runner: api.Runner{Projector: c.opt}}); got != c.want {
				t.Fatalf("ProjectorEnabled = %v, want %v", got, c.want)
			}
		})
	}
}

// With the option off the launch has no projector, so appendMMProjArgs adds
// neither --mmproj nor --no-mmproj-offload and the offload decision is skipped.
func TestProjectorOptionArgs(t *testing.T) {
	gpus := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}}
	for _, c := range []struct {
		name       string
		env        string
		opt        *bool
		projectors []string
		want       []string
		skipped    string
	}{
		{"option on", "", projectorOpt(true), []string{"mmproj.gguf"}, []string{"base", "--mmproj", "mmproj.gguf"}, ""},
		{"unset", "", nil, []string{"mmproj.gguf"}, []string{"base", "--mmproj", "mmproj.gguf"}, ""},
		{"option off", "", projectorOpt(false), []string{"mmproj.gguf"}, []string{"base"}, "mmproj.gguf"},
		{"env off", "1", nil, []string{"mmproj.gguf"}, []string{"base"}, "mmproj.gguf"},
		{"option off, inline projector", "", projectorOpt(false), []string{"model.gguf"}, []string{"base"}, "model.gguf"},
		{"option on beats env off", "1", projectorOpt(true), []string{"mmproj.gguf"}, []string{"base", "--mmproj", "mmproj.gguf"}, ""},
		{"no projector to skip", "1", nil, nil, []string{"base"}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("OLLAMA_NO_PROJECTOR", c.env)
			opts := api.DefaultOptions()
			opts.Projector = c.opt
			projectors, skipped := skipProjector("model.gguf", c.projectors, opts)
			if skipped != c.skipped {
				t.Fatalf("skipped = %q, want %q", skipped, c.skipped)
			}
			got := appendMMProjArgs([]string{"base"}, llamaServerLaunchConfig{
				modelPath:        "model.gguf",
				projectors:       projectors,
				opts:             opts,
				gpus:             gpus,
				mmprojMemory:     933 << 20,
				modelLayers:      81,
				skippedProjector: skipped,
			})
			if !slices.Equal(got, c.want) {
				t.Fatalf("args = %v, want %v", got, c.want)
			}
		})
	}
}

func TestTextOnlyError(t *testing.T) {
	text := &llamaServerRunner{launch: llamaServerLaunchConfig{skippedProjector: "mmproj.gguf"}}
	if !text.TextOnly() {
		t.Fatal("TextOnly = false, want true")
	}
	if err := text.textOnlyError(false, "send images"); err != nil {
		t.Fatalf("no media: unexpected error %v", err)
	}
	err := text.textOnlyError(true, "send images")
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("error = %v, want a 400 StatusError", err)
	}
	for _, want := range []string{`"projector"`, "OLLAMA_NO_PROJECTOR", "text-only"} {
		if !strings.Contains(statusErr.ErrorMessage, want) {
			t.Errorf("error %q does not mention %q", statusErr.ErrorMessage, want)
		}
	}

	vision := &llamaServerRunner{launch: llamaServerLaunchConfig{projectors: []string{"mmproj.gguf"}}}
	if vision.TextOnly() || vision.textOnlyError(true, "send images") != nil {
		t.Fatal("runner with its projector must accept media")
	}
}

// A text-only runner refuses media on every path before contacting llama-server.
func TestTextOnlyRunnerRejectsMedia(t *testing.T) {
	s := &llamaServerRunner{sem: semaphore.NewWeighted(1), launch: llamaServerLaunchConfig{skippedProjector: "mmproj.gguf"}}
	img := api.ImageData("img")

	if _, err := s.llamaServerChatRequest(ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi", Images: []api.ImageData{img}}}}, false); err == nil || !strings.Contains(err.Error(), "projector") {
		t.Errorf("chat: error = %v, want one naming the projector option", err)
	}
	if err := s.Completion(t.Context(), CompletionRequest{Prompt: "[img-0]", Media: []MediaData{NewMediaData(0, img)}}, nil); err == nil || !strings.Contains(err.Error(), "projector") {
		t.Errorf("completion: error = %v, want one naming the projector option", err)
	}
	if _, _, err := s.Embedding(t.Context(), "x", [][]byte{img}); err == nil || !strings.Contains(err.Error(), "projector") {
		t.Errorf("embedding: error = %v, want one naming the projector option", err)
	}
	if _, _, err := s.SystemOne(t.Context(), nil, nil, []api.ImageData{img}); err == nil || !strings.Contains(err.Error(), "projector") {
		t.Errorf("systemone: error = %v, want one naming the projector option", err)
	}
}

// PredictLaunchVRAM leaves out a separate projector file and projector tensors
// stored in the model file when the option is off.
func TestPredictLaunchVRAMProjector(t *testing.T) {
	t.Setenv("OLLAMA_NO_PROJECTOR", "")
	path, f := writeTestGGUF(t, gguftest.KV{"general.architecture": "llama"}, []*gguftest.Tensor{
		testGGUFTensor("token_embd.weight", gguf.TensorTypeF32, []uint64{64}),
		testGGUFTensor("v.blk.0.attn_q.weight", gguf.TensorTypeF32, []uint64{256}),
		testGGUFTensor("mm.0.weight", gguf.TensorTypeF32, []uint64{128}),
	})
	inline := f.Tensors().Size("v.", "mm.", "a.")
	if want := uint64((256 + 128) * 4); inline != want {
		t.Fatalf("inline projector size = %d, want %d", inline, want)
	}
	separate := filepath.Join(t.TempDir(), "mmproj.gguf")
	if err := os.WriteFile(separate, make([]byte, 1000), 0o644); err != nil {
		t.Fatal(err)
	}
	off := api.Options{Runner: api.Runner{Projector: projectorOpt(false)}}
	on := api.Options{Runner: api.Runner{Projector: projectorOpt(true)}}

	base := PredictServerVRAM(path, f, 4096, nil)
	for name, c := range map[string]struct {
		projectors []string
		opts       api.Options
		want       uint64
	}{
		"on, inline":     {[]string{path}, on, base},
		"off, inline":    {[]string{path}, off, base - inline},
		"on, separate":   {[]string{separate}, on, base + 1000},
		"off, separate":  {[]string{separate}, off, base - inline},
		"off, no mmproj": {nil, off, base - inline},
	} {
		if got := PredictLaunchVRAM(path, f, 4096, c.projectors, c.opts); got != c.want {
			t.Errorf("%s: PredictLaunchVRAM = %d, want %d", name, got, c.want)
		}
	}
}
