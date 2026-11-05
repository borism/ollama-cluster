package cluster

import (
	"fmt"
	"log/slog"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/format"
)

// localCPUShare estimates the share of a model that a local-only load leaves
// on the CPU: the shortfall (predicted minus what is free on the local GPUs,
// the quantity SelectRPCServers spills) as a fraction of predicted. llama.cpp
// moves whole layers, weights and their KV together, so for layers of equal
// size this is also the share of layers, and of weights, on the CPU: 2 of 65
// layers is about 0.03.
//
// It leaves out fillMargin on purpose. That margin is for a computed
// --tensor-split, which turns llama.cpp's --fit off; a local-only load keeps
// --fit, which sizes the real compute buffers itself. With the margin, any
// model that misses a 20 GiB GPU would count as 15 % or more on the CPU, yet
// the models in #73 that spilled left only 2-3 layers there.
func localCPUShare(local, predicted uint64) (shortfall uint64, share float64) {
	if predicted == 0 || local >= predicted {
		return 0, 0
	}
	shortfall = predicted - local
	return shortfall, float64(shortfall) / float64(predicted)
}

// keepLocal reports whether SelectRPCServers should load a model that misses
// the local GPUs only by a little without peers, leaving those few layers on
// the CPU (#73): local-only was as fast at generation and about twice as fast
// at prompt processing as the cluster for models 2-3 layers short. The limit
// is envconfig.ClusterMinSpill; at 0 (the default) it never does.
//
// A request that asks for the cluster itself, with rpc_auto true or its own
// rpc_placement, is spilled as before: the threshold is a server-wide default
// and does not override a per-request choice. The server-wide
// OLLAMA_CLUSTER_PLACEMENT only picks the policy among peers, so it does not
// count as asking. rpc_servers never gets here (SelectRPCServers returns it
// as is).
func keepLocal(local, predicted uint64, opts api.Options) bool {
	limit := envconfig.ClusterMinSpill()
	if limit <= 0 {
		return false
	}
	shortfall, share := localCPUShare(local, predicted)
	if shortfall == 0 || share > limit {
		return false
	}
	attrs := []any{
		"predicted", format.HumanBytes2(predicted), "local", format.HumanBytes2(local),
		"shortfall", format.HumanBytes2(shortfall), "cpu_share", fmt.Sprintf("%.3f", share), "min_spill", limit,
	}
	if (opts.RPCAuto != nil && *opts.RPCAuto) || opts.RPCPlacement != "" {
		slog.Debug("cluster: shortfall is under OLLAMA_CLUSTER_MIN_SPILL, but the request asked for the cluster (rpc_auto or rpc_placement), spilling", attrs...)
		return false
	}
	slog.Info("cluster: shortfall is under OLLAMA_CLUSTER_MIN_SPILL, loading local-only with it on the CPU", attrs...)
	return true
}
