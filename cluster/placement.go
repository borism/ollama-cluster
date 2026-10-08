package cluster

import (
	"fmt"
	"log/slog"
	"math"
	"net"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/ml"
)

// RPCProtoMajor/RPCProtoMinor are defined in discovery.go.

// minUsefulMemory: a peer with less spare memory than this isn't worth
// an RPC hop for.
const minUsefulMemory = 1024 * 1024 * 1024 // 1 GiB

// localAvailable is the same "usable free memory" convention
// server/sched.go's load() applies per GPU before deciding placement
// (envconfig.GpuOverhead() + gpu.MinimumMemory() set aside, floored at 0).
func localAvailable(gpus []ml.DeviceInfo) uint64 {
	var total uint64
	overhead := envconfig.GpuOverhead()
	for _, gpu := range gpus {
		reserve := overhead + gpu.MinimumMemory()
		if gpu.FreeMemory > reserve {
			total += gpu.FreeMemory - reserve
		}
	}
	return total
}

// peerFreeMemory is a peer's spillable capacity: the sum of its
// self-reported devices' free memory. No overhead margin -- that's the
// peer's own ggml-rpc-server's problem when it actually allocates.
func peerFreeMemory(p Peer) uint64 {
	var total uint64
	for _, d := range p.Devices {
		total += d.FreeMemory
	}
	return total
}

func usablePeer(p Peer) bool {
	return p.RPCPort != 0 && p.ProtoMajor == RPCProtoMajor && p.ProtoMinor <= RPCProtoMinor &&
		peerFreeMemory(p) >= minUsefulMemory
}

// waterFill picks frac_i minimizing the shared completion time
// max_i(frac_i/rates[i] + links[i]), subject to sum(frac_i) <= 1 and
// 0 <= frac_i <= caps[i] -- standard water-filling for a divisible load
// across heterogeneous workers, solved by bisection on T: frac_i(T) =
// clamp(rates[i]*(T-links[i]), 0, caps[i]) is monotonically
// non-decreasing in T, so bisect for the T where the sum hits 1 (or, if
// total capacity can't reach 1, converges with every peer maxed at its
// own cap -- the correct "everyone helps as much as they can" answer).
func waterFill(rates, links, caps []float64) []float64 {
	fracsAt := func(t float64) []float64 {
		out := make([]float64, len(rates))
		for i := range rates {
			out[i] = math.Max(0, math.Min(caps[i], rates[i]*(t-links[i])))
		}
		return out
	}

	hi := 0.0
	for _, l := range links {
		hi = math.Max(hi, l)
	}
	for i := range rates {
		if rates[i] > 0 {
			hi = math.Max(hi, caps[i]/rates[i])
		}
	}
	hi += 1.0

	lo := 0.0
	for range 60 {
		mid := (lo + hi) / 2
		sum := 0.0
		for _, f := range fracsAt(mid) {
			sum += f
		}
		if sum < 1.0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	return fracsAt(hi)
}

// SelectRPCServers decides whether opts should spill onto cluster peers to
// fit a model whose predicted VRAM need is `predicted`, and if so fills in
// opts.RPCServers (see api.Runner.RPCServers, llm/llama_server.go's
// appendRPCArgs -- this function only needs to produce that string, nothing
// downstream needs to change). Pure function: no network calls -- peer
// link latency arrives pre-measured on Peer.Latency (see discovery.go's
// probeLoop), so this is just a decision over the snapshot it's given.
//
// Two selection policies (opts.RPCPlacement, falling back to
// envconfig.ClusterPlacement()):
//   - "waterfill" (default, empty also means this): peers are weighted by
//     self-reported headroom (1-Load) and measured link latency, not just
//     free memory, and every peer with a positive share ends up in the
//     list (splitting a shortfall across several half-free peers can beat
//     parking it all on the one biggest peer).
//   - "greedy": the original v1 behavior -- peers ranked by free memory,
//     added until the shortfall is covered, ignoring load/latency. Uses
//     the fewest peers (and network hops) rather than balancing load;
//     kept as an option for comparison or when minimizing hops matters
//     more than balancing.
//
// Either way, the actual byte split across included servers is still
// llama.cpp's own job (automatic, proportional to each device's reported
// free memory) unless opts.TensorSplit overrides it manually
// (llm/llama_server.go's appendRPCArgs); this function only decides which
// peers are worth the hop and in what priority order. The "waterfill"
// rate is a 0..1 headroom score, not a measured tokens/sec -- a real
// per-device throughput bench is still a fast-follow.
func SelectRPCServers(gpus []ml.DeviceInfo, predicted uint64, peers []Peer, opts api.Options) api.Options {
	if opts.RPCServers != "" {
		return opts
	}

	local := localAvailable(gpus)
	if local >= predicted {
		return opts
	}

	usable := make([]Peer, 0, len(peers))
	for _, p := range peers {
		if usablePeer(p) {
			usable = append(usable, p)
		}
	}
	if len(usable) == 0 {
		return opts
	}

	policy := opts.RPCPlacement
	if policy == "" {
		policy = envconfig.ClusterPlacement()
	}

	var addrs []string
	if policy == "greedy" {
		addrs = selectGreedy(usable, local, predicted)
	} else {
		addrs = selectWaterFill(usable, predicted-local)
	}

	opts.RPCServers = strings.Join(addrs, ",")
	return opts
}

// selectGreedy: peers ranked by free memory descending, added until the
// running total (starting from local capacity) covers predicted. Stops as
// soon as it's covered, so a peer not needed for that stays unused even if
// idle -- see SelectRPCServers's doc comment for the water-fill contrast.
func selectGreedy(usable []Peer, local, predicted uint64) []string {
	sorted := append([]Peer(nil), usable...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return peerFreeMemory(sorted[i]) > peerFreeMemory(sorted[j])
	})

	cum := local
	addrs := make([]string, 0, len(sorted))
	for _, p := range sorted {
		if cum >= predicted {
			break
		}
		addrs = append(addrs, rpcAddr(p))
		cum += peerFreeMemory(p)
	}
	return addrs
}

// selectWaterFill weighs peers by self-reported headroom and measured
// link latency via waterFill, includes every peer with a positive share,
// and orders the result by share descending.
func selectWaterFill(usable []Peer, shortfall uint64) []string {
	rates := make([]float64, len(usable))
	links := make([]float64, len(usable))
	caps := make([]float64, len(usable))
	for i, p := range usable {
		// Floored so a fully-busy peer still gets a small share instead of
		// a zero rate that a network hiccup could then divide by.
		rates[i] = math.Max(0.05, 1-p.Load)
		links[i] = p.Latency.Seconds()
		caps[i] = float64(peerFreeMemory(p)) / float64(shortfall)
	}
	fracs := waterFill(rates, links, caps)

	type share struct {
		peer Peer
		frac float64
	}
	picked := make([]share, 0, len(usable))
	for i, p := range usable {
		if fracs[i] > 0 {
			picked = append(picked, share{p, fracs[i]})
		}
	}
	sort.SliceStable(picked, func(i, j int) bool {
		return picked[i].frac > picked[j].frac
	})

	addrs := make([]string, len(picked))
	for i, s := range picked {
		addrs[i] = rpcAddr(s.peer)
	}
	return addrs
}

// rpcAddr is how a peer's ggml-rpc-server appears in --rpc.
func rpcAddr(p Peer) string {
	return fmt.Sprintf("%s:%d", p.Addr, p.RPCPort)
}

// SelectReachableRPCServers is SelectRPCServers, but checks that every peer
// it picks accepts a connection on its RPC port first, and picks again
// without the ones that don't. llama-server aborts the whole load when it
// can't reach an --rpc server (ggml-rpc.cpp "Failed to connect"), so a peer
// that is still beaconing but blocked (firewall, macOS Local Network
// permission) or just gone would otherwise stop the model loading at all.
// With no reachable peer left it returns no RPCServers, and the model loads
// locally. RPCServers the caller set itself are passed through unchecked.
//
// dial is DialRPC in production; it is a parameter so tests don't need real
// listeners. It runs while the scheduler holds its load lock, which is why
// it's bounded by probeTimeout.
func SelectReachableRPCServers(gpus []ml.DeviceInfo, predicted uint64, peers []Peer, opts api.Options, dial func(addr string) error) api.Options {
	if opts.RPCServers != "" {
		return opts
	}
	for {
		picked := SelectRPCServers(gpus, predicted, peers, opts)
		if picked.RPCServers == "" {
			return picked
		}

		addrs := strings.Split(picked.RPCServers, ",")
		errs := make([]error, len(addrs))
		var wg sync.WaitGroup
		for i, addr := range addrs {
			wg.Go(func() { errs[i] = dial(addr) })
		}
		wg.Wait()

		unreachable := map[string]bool{}
		for i, err := range errs {
			if err != nil {
				slog.Warn("cluster: skipping unreachable peer", "addr", addrs[i], "error", err)
				unreachable[addrs[i]] = true
			}
		}
		if len(unreachable) == 0 {
			return picked
		}
		// Terminates: every round removes at least one peer.
		peers = slices.DeleteFunc(slices.Clone(peers), func(p Peer) bool { return unreachable[rpcAddr(p)] })
	}
}

// DialRPC checks that addr accepts a TCP connection within probeTimeout,
// the same check probeLoop times for Peer.Latency.
func DialRPC(addr string) error {
	conn, err := net.DialTimeout("tcp", addr, probeTimeout)
	if err != nil {
		return err
	}
	return conn.Close()
}
