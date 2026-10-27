package main

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/app/server"
	"github.com/borism/ollama-cluster/envconfig"
)

// The menu-bar items show and change the same cluster settings as the
// Settings page and `ollama cluster`: the app's `ollama serve` owns them
// (GET/POST /api/cluster/config, saved in ~/.ollama/server.json).
var clusterMenu server.ClusterMenuCache

// clusterMenuState returns the settings to show: what the server last
// said, which can lag behind it. ok is false until the server has answered
// once (e.g. still starting). Never waits on the server, as the menu draws
// it on the main thread; see refreshClusterMenu.
func clusterMenuState() (cfg api.ClusterConfig, ok bool) {
	clusterMenu.Lock()
	defer clusterMenu.Unlock()
	return clusterMenu.Cfg, clusterMenu.Known
}

// refreshClusterMenu asks the server for the settings, unless a change is
// still being applied. It can take a while (GET /api/cluster/config
// re-queries the GPUs), so call it off the main thread. If the server
// doesn't answer, the menu keeps showing the last settings.
func refreshClusterMenu() {
	clusterMenu.Lock()
	busy := clusterMenu.Pending > 0
	clusterMenu.Unlock()
	if busy {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	fresh, err := api.NewClient(envconfig.Host(), http.DefaultClient).ClusterConfig(ctx)
	if err != nil {
		slog.Debug("cluster menu: server not reachable", "error", err)
	}

	clusterMenu.Lock()
	defer clusterMenu.Unlock()
	clusterMenu.Apply(fresh, err)
}

// updateClusterConfig shows a change in the menu right away and sends it
// to the server in the background: turning cluster mode or sharing on
// waits for the RPC worker to start (the first time on Apple Silicon, it
// compiles Metal kernels for ~20s), and menu actions run on the main
// thread. If the server rejects it, the next menu open shows the real
// state again.
func updateClusterConfig(req api.ClusterConfigRequest, show func(*api.ClusterConfig)) {
	clusterMenu.Lock()
	show(&clusterMenu.Cfg)
	clusterMenu.Pending++
	clusterMenu.Unlock()

	go func() {
		cfg, err := api.NewClient(envconfig.Host(), http.DefaultClient).UpdateClusterConfig(context.Background(), &req)
		clusterMenu.Lock()
		defer clusterMenu.Unlock()
		clusterMenu.Pending--
		if err != nil {
			slog.Error("failed to change cluster settings", "error", err)
		}
		clusterMenu.Apply(cfg, err)
	}()
}
