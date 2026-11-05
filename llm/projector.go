package llm

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/fs/gguf"
)

// The "projector" option (api.Runner.Projector, OLLAMA_NO_PROJECTOR) loads a
// model text-only: llama-server starts without --mmproj, so it never
// initializes mtmd and the projector's weights and worst-case encode buffers
// take no memory. This works for a separate projector file and for one
// stored inside the model file alike: llama/compat hides inline projector
// tensors (v.*, mm.*, a.*) from the text loader whether or not --mmproj is
// passed, so without it nothing reads them.

// ProjectorEnabled reports whether a launch with opts loads the model's
// multimodal projector: the "projector" option when set, otherwise the
// server's OLLAMA_NO_PROJECTOR default.
func ProjectorEnabled(opts api.Options) bool {
	if opts.Projector != nil {
		return *opts.Projector
	}
	return !envconfig.NoProjector()
}

// skipProjector drops the projector from a launch whose projector option is
// off, returning the projectors to load and the one skipped, if any.
func skipProjector(modelPath string, projectors []string, opts api.Options) ([]string, string) {
	if len(projectors) == 0 || ProjectorEnabled(opts) {
		return projectors, ""
	}
	reason := `"projector": false`
	if opts.Projector == nil {
		reason = "OLLAMA_NO_PROJECTOR"
	}
	slog.Info("loading model text-only, without its multimodal projector",
		"model", modelPath, "projector", projectors[0], "inline", projectors[0] == modelPath, "reason", reason)
	return nil, projectors[0]
}

// TextOnly reports whether the runner was started without the model's
// projector because the projector option was off.
func (s *llamaServerRunner) TextOnly() bool {
	return s.launch.skippedProjector != ""
}

// textOnlyError rejects a request that needs the projector a text-only
// runner skipped, naming the option, instead of letting llama-server fail
// on (or ignore) the media.
func (s *llamaServerRunner) textOnlyError(needed bool, use string) error {
	if !needed || !s.TextOnly() {
		return nil
	}
	return api.StatusError{
		StatusCode: http.StatusBadRequest,
		ErrorMessage: fmt.Sprintf(`model was loaded text-only, without its multimodal projector ("projector": false, or OLLAMA_NO_PROJECTOR=1 on the server); set the "projector" option to true to %s`,
			use),
	}
}

// PredictLaunchVRAM is PredictServerVRAM for what a launch with opts loads:
// with the projector option off it counts neither a separate projector file
// nor projector tensors stored in the model file, which the text loader
// never reads.
func PredictLaunchVRAM(modelPath string, f *gguf.Model, numCtx int, projectors []string, opts api.Options) uint64 {
	if ProjectorEnabled(opts) {
		return PredictServerVRAM(modelPath, f, numCtx, projectors)
	}
	vram := PredictServerVRAM(modelPath, f, numCtx, nil)
	return vram - min(f.Tensors().Size("v.", "mm.", "a."), vram)
}
