package server

import (
	"cmp"
	"context"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/cluster"
	"github.com/borism/ollama-cluster/discover"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/llm"
	"github.com/borism/ollama-cluster/ml"
)

// clusterRunner owns the running cluster subsystem -- discovery and, when
// sharing, the RPC worker -- so cluster mode can be switched while the
// server runs (`ollama cluster on/off/set`, POST /api/cluster/config)
// instead of only at startup.
type clusterRunner struct {
	ctx   context.Context //nolint:containedctx
	sched *Scheduler
	// start is startCluster; a field so tests can count starts and stops
	// without binding real ports.
	start func(ctx context.Context, sched *Scheduler) (stop func())

	mu      sync.Mutex
	stop    func() // nil while cluster mode is off
	running clusterRunSettings
}

// clusterRunSettings are the settings that only take effect when the
// subsystem starts. Placement isn't one: it's read on every model load.
type clusterRunSettings struct {
	share   bool
	seeds   string
	devices string
	cacheGB uint
}

func newClusterRunner(ctx context.Context, sched *Scheduler) *clusterRunner {
	return &clusterRunner{ctx: ctx, sched: sched, start: startCluster}
}

// apply brings the running subsystem in line with the current settings
// (envconfig.Cluster and friends): starts it, stops it, or restarts it if a
// setting it was started with changed. Otherwise it's left running, so
// peers using this machine's worker aren't cut off for nothing.
func (r *clusterRunner) apply() {
	r.mu.Lock()
	defer r.mu.Unlock()

	want := clusterRunSettings{
		share:   envconfig.ClusterShare(true),
		seeds:   strings.Join(envconfig.ClusterSeeds(), ","),
		devices: strings.Join(envconfig.ClusterShareDevices(), ","),
		cacheGB: envconfig.ClusterCacheGB(),
	}
	enabled := envconfig.Cluster()
	if r.stop != nil && enabled && want == r.running {
		return
	}
	if r.stop != nil {
		slog.Info("cluster: stopping")
		r.stop()
		r.stop = nil
	}
	if enabled {
		slog.Info("cluster: starting", "share", want.share, "seeds", want.seeds)
		r.stop = r.start(r.ctx, r.sched)
		r.running = want
	}
}

// filterSharedDevices keeps the devices whose ggml name (ml.DeviceInfo.Name,
// e.g. "CUDA1", the same name rpc-server's -d takes) is in shared. An empty
// shared list keeps everything.
func filterSharedDevices(devs []ml.DeviceInfo, shared []string) []ml.DeviceInfo {
	if len(shared) == 0 {
		return devs
	}
	var out []ml.DeviceInfo
	for _, d := range devs {
		if slices.ContainsFunc(shared, func(s string) bool { return strings.EqualFold(s, d.Name) }) {
			out = append(out, d)
		}
	}
	return out
}

// startCluster starts LAN peer discovery (cluster.Start) and, unless
// sharing is off, this instance's RPC worker (llm.StartRPCWorker). The
// returned stop shuts both down and waits until discovery's UDP port is
// free again.
func startCluster(parent context.Context, sched *Scheduler) (stop func()) {
	ctx, cancel := context.WithCancel(parent)

	var worker *llm.RPCWorker
	rpcPort := func() int { return 0 }
	if envconfig.ClusterShare(true) {
		var err error
		worker, err = llm.StartRPCWorker(0, envconfig.Models(), envconfig.ClusterShareDevices())
		if err != nil {
			slog.Warn("cluster: failed to start RPC worker, this instance will not share GPU capacity", "error", err)
		} else {
			rpcPort = worker.Port
			go func() {
				<-ctx.Done()
				worker.Stop()
			}()
		}
	}

	table, err := cluster.Start(ctx, cluster.Config{
		Port:     int(envconfig.ClusterPort()),
		Interval: 5 * time.Second,
		TTL:      15 * time.Second,
		SelfDevices: func() []ml.DeviceInfo {
			return filterSharedDevices(discover.GPUDevices(ctx, nil), envconfig.ClusterShareDevices())
		},
		SelfRPCPort: rpcPort,
		SelfLoad:    sched.clusterLoad,
		Seeds:       envconfig.ClusterSeeds(),
	})
	if err != nil {
		slog.Warn("cluster: failed to start discovery", "error", err)
		cancel()
		if worker != nil {
			worker.Stop()
		}
		return func() {}
	}
	sched.clusterTable.Store(table)
	go logClusterPeers(ctx, table)

	return func() {
		sched.clusterTable.Store(nil)
		cancel()
		<-table.Stopped()
		if worker != nil {
			worker.Stop()
		}
	}
}

// logClusterPeers is operability, not correctness: cluster.Table has no
// on-change hook, so this just polls Peers() -- fine at a 10s cadence, only
// running while cluster mode is on.
func logClusterPeers(ctx context.Context, table *cluster.Table) {
	seen := map[string]bool{}
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := map[string]bool{}
			for _, p := range table.Peers() {
				now[p.ID] = true
				if !seen[p.ID] {
					slog.Info("cluster: peer discovered", "id", p.ID, "addr", p.Addr, "rpc_port", p.RPCPort, "devices", len(p.Devices))
				}
			}
			for id := range seen {
				if !now[id] {
					slog.Info("cluster: peer expired", "id", id)
				}
			}
			seen = now
		}
	}
}

// localGPUDevices lists this machine's GPUs; a variable so tests can fake it.
var localGPUDevices = func(ctx context.Context) []ml.DeviceInfo { return discover.GPUDevices(ctx, nil) }

// currentClusterConfig is what GET and POST /api/cluster/config return.
func currentClusterConfig(ctx context.Context) api.ClusterConfig {
	var devices []api.ClusterDevice
	for _, d := range localGPUDevices(ctx) {
		devices = append(devices, api.ClusterDevice{Name: d.Name, Description: d.Description, TotalMemory: d.TotalMemory})
	}
	return api.ClusterConfig{
		Enabled:        envconfig.Cluster(),
		Share:          envconfig.ClusterShare(true),
		Seeds:          strings.Join(envconfig.ClusterSeeds(), ","),
		ShareDevices:   strings.Join(envconfig.ClusterShareDevices(), ","),
		Placement:      cmp.Or(envconfig.ClusterPlacement(), "waterfill"),
		CacheGB:        envconfig.ClusterCacheGB(),
		CacheUsedBytes: llm.RPCCacheBytes(envconfig.Models()),
		Sources:        envconfig.ClusterSources(),
		Devices:        devices,
	}
}

// ClusterConfigHandler returns this instance's cluster-mode settings.
func (s *Server) ClusterConfigHandler(c *gin.Context) {
	c.JSON(http.StatusOK, currentClusterConfig(c.Request.Context()))
}

// UpdateClusterConfigHandler saves changed cluster settings in this
// server's ~/.ollama/server.json and applies them right away. Writing the
// file here rather than in the CLI means it's the file the server actually
// reads, whichever user the service runs as (the Linux systemd unit runs as
// "ollama", home /usr/share/ollama).
//
// Only accepted from a loopback address: cluster mode offers this
// machine's GPU to the whole LAN, unauthenticated (docs/cluster.mdx,
// "Security"), so a client that can merely reach a server exposed with
// OLLAMA_HOST=0.0.0.0 mustn't be able to turn it on. A reverse proxy on the
// same machine would count as local.
func (s *Server) UpdateClusterConfigHandler(c *gin.Context) {
	if !requestFromLoopback(c.Request) {
		c.JSON(http.StatusForbidden, gin.H{"error": "cluster settings can only be changed from the machine the server runs on"})
		return
	}

	var req api.ClusterConfigRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	set := map[string]any{}
	if req.Enabled != nil {
		set["cluster"] = *req.Enabled
	}
	if req.Share != nil {
		set["cluster_share"] = *req.Share
	}
	if req.Seeds != nil {
		if err := cluster.ValidateSeeds(*req.Seeds); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		set["cluster_seeds"] = strings.TrimSpace(*req.Seeds)
	}
	if req.ShareDevices != nil {
		set["cluster_share_devices"] = strings.TrimSpace(*req.ShareDevices)
	}
	if req.Placement != nil {
		if err := cluster.ValidatePlacement(*req.Placement); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		set["cluster_placement"] = *req.Placement
	}
	if req.CacheGB != nil {
		set["cluster_cache_gb"] = *req.CacheGB
	}
	if len(set) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no cluster setting given"})
		return
	}

	if err := envconfig.UpdateServerConfig(set); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if s.cluster != nil {
		s.cluster.apply()
	}
	c.JSON(http.StatusOK, currentClusterConfig(c.Request.Context()))
}

// requestFromLoopback reports whether r's TCP peer is a loopback address.
// Deliberately RemoteAddr, not gin's ClientIP: that trusts X-Forwarded-For
// by default, which any remote client can set.
func requestFromLoopback(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
