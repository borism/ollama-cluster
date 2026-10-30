package cluster

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/format"
	"github.com/borism/ollama-cluster/ml"
)

// RPCProtoMajor/RPCProtoMinor are defined in rpcproto_gen.go.

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
	return p.RPCPort != 0 && p.ProtoIncompatibility() == "" &&
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
// Either way the policies only nominate candidates. With one local GPU and
// no opts.TensorSplit, fillSplit then lays the model out local-first and
// keeps only the peers that get weights, setting opts.TensorSplit; otherwise
// the byte split is llama.cpp's own (automatic, proportional to each
// device's reported free memory). The "waterfill" rate is a 0..1 headroom
// score, not a measured tokens/sec.
func SelectRPCServers(gpus []ml.DeviceInfo, predicted uint64, peers []Peer, opts api.Options) api.Options {
	if opts.RPCServers != "" {
		return opts
	}
	opts = ignoreTensorSplit(opts)

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

	policy := opts.RPCPlacement
	if policy == "" {
		policy = envconfig.ClusterPlacement()
	}
	// One line per decision, so a peer that was left out can be explained.
	defer func() {
		slog.Info("cluster: placement", "predicted", format.HumanBytes2(predicted), "local", format.HumanBytes2(local),
			"policy", policy, "candidates", describePeers(peers, time.Now()),
			"rpc_servers", opts.RPCServers, "tensor_split", opts.TensorSplit)
	}()
	if len(usable) == 0 {
		return opts
	}

	var addrs []string
	if policy == "greedy" {
		addrs = selectGreedy(usable, local, predicted)
	} else {
		addrs = selectWaterFill(usable, predicted-local)
	}

	opts.RPCServers = strings.Join(addrs, ",")
	if opts.TensorSplit == "" {
		if picked, split, ok := fillSplit(gpus, predicted, peersAt(usable, addrs)); ok {
			opts.RPCServers = strings.Join(picked, ",")
			opts.TensorSplit = split
		}
	}
	return opts
}

// describePeers is one log field naming every peer placement was handed with
// its free memory, rpc_port, whether it was usable (and if not, why) and how
// long ago its last beacon was.
func describePeers(peers []Peer, now time.Time) string {
	if len(peers) == 0 {
		return "none"
	}
	parts := make([]string, len(peers))
	for i, p := range peers {
		verdict := "usable"
		switch {
		case p.RPCPort == 0:
			verdict = "not sharing (rpc_port 0)"
		case p.ProtoIncompatibility() != "":
			verdict = p.ProtoIncompatibility()
		case peerFreeMemory(p) < minUsefulMemory:
			verdict = "under 1 GiB free"
		}
		age := "unknown"
		if !p.LastSeen.IsZero() {
			age = now.Sub(p.LastSeen).Round(100 * time.Millisecond).String()
		}
		parts[i] = fmt.Sprintf("%s free=%s rpc_port=%d proto=%d.%d beacon_age=%s: %s",
			p.Addr, format.HumanBytes2(peerFreeMemory(p)), p.RPCPort, p.ProtoMajor, p.ProtoMinor, age, verdict)
	}
	return strings.Join(parts, "; ")
}

// ignoreTensorSplit drops a caller's TensorSplit when the caller did not also
// choose RPCServers. llama.cpp maps its entries by device position (peers in
// --rpc order, then the local GPU), and the peers the server picks, and their
// order, change with the policy and with each peer's load, so the same split
// would land on different devices from one request to the next (a "4,6" meant
// for a peer and the GPU put 60 % on the second peer and nothing on the GPU).
// It is not applied to a local-only load either (appendRPCArgs).
func ignoreTensorSplit(opts api.Options) api.Options {
	if opts.RPCServers == "" && opts.TensorSplit != "" {
		slog.Warn("cluster: ignoring tensor_split because rpc_servers is not set; it maps to devices by position, which automatic peer selection does not fix",
			"tensor_split", opts.TensorSplit)
		opts.TensorSplit = ""
	}
	return opts
}

// peersAt returns the peers of usable whose --rpc address is in addrs.
func peersAt(usable []Peer, addrs []string) []Peer {
	return slices.DeleteFunc(slices.Clone(usable), func(p Peer) bool { return !slices.Contains(addrs, rpcAddr(p)) })
}

// fillMargin is what a device keeps back from fillSplit for memory the split
// does not count: the CUDA/Metal context, compute buffers and the vision
// projector. Measured on a 20 GiB RTX A4500 (docs/cluster-benchmarks.mdx):
// everything but weights and KV came to 0.5-2.6 GiB.
const fillMargin = 3 << 30

// fillSplit lays the model out for speed instead of for free memory: the
// local GPU takes as much as fits, then the peers in order of free memory
// (largest first, a stand-in for speed), and a peer that is not needed gets
// nothing and is left out of --rpc. It returns the peers kept, in --rpc order,
// and the matching --tensor-split (peers first, then the local GPU -- see
// api.Runner.TensorSplit). llama.cpp's own split is proportional to free
// memory, which puts weights on a slow peer just because it has room: on a
// 3-device fleet that cost 2x (docs/cluster-benchmarks.mdx).
//
// ok is false -- keep llama.cpp's split and its --fit -- when the layout isn't
// simple: no GPU or more than one local (their order and library are
// llama.cpp's call), an integrated GPU (not faster than a peer), or a model
// that doesn't fit the devices at all (--fit then spills to CPU, and an
// explicit split would turn that off).
func fillSplit(gpus []ml.DeviceInfo, need uint64, peers []Peer) (kept []string, split string, ok bool) {
	if len(gpus) != 1 || gpus[0].Integrated {
		return nil, "", false
	}
	// ponytail: free memory as the speed rank; a bandwidth table or a measured
	// bench per device name would order a fast small peer before a slow big one.
	ranked := slices.Clone(peers)
	sort.SliceStable(ranked, func(i, j int) bool { return peerFreeMemory(ranked[i]) > peerFreeMemory(ranked[j]) })

	room := func(free uint64) uint64 {
		if free > fillMargin {
			return free - fillMargin
		}
		return 0
	}
	left := need
	take := func(room uint64) uint64 {
		n := min(left, room)
		left -= n
		return n
	}
	localTake := take(room(localAvailable(gpus)))
	var shares []string
	for _, p := range ranked {
		if n := take(room(peerFreeMemory(p))); n > 0 {
			kept = append(kept, rpcAddr(p))
			shares = append(shares, strconv.FormatUint(n>>20, 10))
		}
	}
	if left > 0 {
		return nil, "", false
	}
	shares = append(shares, strconv.FormatUint(localTake>>20, 10))
	return kept, strings.Join(shares, ","), true
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
// it picks answers the RPC handshake (DialRPC) first, and picks again
// without the ones that don't. llama-server aborts the whole load when it
// can't reach an --rpc server (ggml-rpc.cpp "Failed to connect"), so a peer
// that is still beaconing but blocked (firewall, macOS Local Network
// permission) or just gone would otherwise stop the model loading at all.
// With no reachable peer left it returns no RPCServers, and the model loads
// locally. RPCServers the caller set itself are passed through unchecked.
//
// dial is DialRPC in production; it is a parameter so tests don't need real
// listeners. It runs while the scheduler holds its load lock, which is why
// it's bounded by probeTimeout plus rpcHelloTimeout.
func SelectReachableRPCServers(gpus []ml.DeviceInfo, predicted uint64, peers []Peer, opts api.Options, dial func(addr string) error) api.Options {
	if opts.RPCServers != "" {
		return opts
	}
	opts = ignoreTensorSplit(opts) // once, not on every round below
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

// rpcHelloTimeout bounds DialRPC's whole exchange. A worker that is up answers
// in well under a millisecond; one that is busy with another client (the
// ggml-rpc-server accept loop serves one connection at a time, see
// rpc_serve_client in ggml/src/ggml-rpc/ggml-rpc.cpp) never does.
const rpcHelloTimeout = 2 * time.Second

// rpcConnCapsSize is RPC_CONN_CAPS_SIZE (ggml/src/ggml-rpc/transport.h).
const rpcConnCapsSize = 24

// rpcCmdHello is RPC_CMD_HELLO, pinned to 14 by a static_assert in ggml-rpc.cpp.
const rpcCmdHello = 14

// DialRPC checks that the ggml-rpc-server at addr is up and speaks our
// protocol: it connects within probeTimeout and then does the RPC HELLO
// exchange (client side: negotiate_hello in ggml-rpc.cpp). A bare TCP connect
// is not enough -- a worker can accept the connection and reset it right away,
// and llama-server then aborts the load in ggml_backend_rpc_add_server with an
// error that does not name the peer. Sending no transport capabilities (all
// zero) keeps the connection plain TCP; the worker just reads HELLO, answers
// and sees us hang up.
func DialRPC(addr string) error {
	conn, err := net.DialTimeout("tcp", addr, probeTimeout)
	if err != nil {
		return err
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(rpcHelloTimeout)); err != nil {
		return err
	}

	// | cmd (1) | size (8) | rpc_msg_hello_req: conn_caps (24) |
	req := make([]byte, 1+8+rpcConnCapsSize)
	req[0] = rpcCmdHello
	binary.LittleEndian.PutUint64(req[1:], rpcConnCapsSize)
	if _, err := conn.Write(req); err != nil {
		return fmt.Errorf("rpc hello: %w", err)
	}

	// | size (8) | rpc_msg_hello_rsp: major, minor, patch, padding (4) + conn_caps (24) |
	rsp := make([]byte, 8+4+rpcConnCapsSize)
	if _, err := io.ReadFull(conn, rsp); err != nil {
		return fmt.Errorf("rpc hello: %w", err)
	}
	if size := binary.LittleEndian.Uint64(rsp); size != 4+rpcConnCapsSize {
		return fmt.Errorf("rpc hello: unexpected reply size %d", size)
	}
	if major, minor := int(rsp[8]), int(rsp[9]); major != RPCProtoMajor || minor > RPCProtoMinor {
		return fmt.Errorf("rpc hello: worker speaks protocol %d.%d, this node needs %d.%d", major, minor, RPCProtoMajor, RPCProtoMinor)
	}
	return nil
}
