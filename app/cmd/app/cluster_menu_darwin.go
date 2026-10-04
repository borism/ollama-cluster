package main

// #include <stdbool.h>
import "C"

import (
	"strings"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/app/server"
)

// Exported to cluster_menu_darwin.m, which draws the cluster-mode items in
// the menu-bar menu. See cluster_settings.go for where the settings live.

// ClusterMenuState fills in the settings the menu shows and whether each is
// locked by an OLLAMA_CLUSTER* environment variable on the server (the
// menu can't change those). Returns false while the server isn't reachable.
//
//export ClusterMenuState
func ClusterMenuState(enabled, share, greedy, enabledLocked, shareLocked, placementLocked *C.bool) C.bool {
	cfg, ok := clusterMenuState()
	*enabled = C.bool(cfg.Enabled)
	*share = C.bool(cfg.Share)
	*greedy = C.bool(cfg.Placement == "greedy")
	*enabledLocked = C.bool(cfg.Sources["enabled"] == "env")
	*shareLocked = C.bool(cfg.Sources["share"] == "env" || cfg.Sources["share_devices"] == "env")
	*placementLocked = C.bool(cfg.Sources["placement"] == "env")
	return C.bool(ok)
}

//export SetClusterModeEnabled
func SetClusterModeEnabled(enabled C.bool) {
	b := bool(enabled)
	updateClusterConfig(api.ClusterConfigRequest{Enabled: &b}, func(c *api.ClusterConfig) { c.Enabled = b })
}

// ToggleClusterShare flips the share switch for the GPU called name ("" for
// the CPU-only switch), saving share and share_devices together.
//
//export ToggleClusterShare
func ToggleClusterShare(name *C.char) {
	clusterMenu.Lock()
	share, devices := server.ClusterShareToggle(clusterMenu.cfg, C.GoString(name))
	clusterMenu.Unlock()
	updateClusterConfig(api.ClusterConfigRequest{Share: &share, ShareDevices: &devices}, func(c *api.ClusterConfig) {
		c.Share, c.ShareDevices = share, devices
	})
}

//export SetClusterPlacementGreedy
func SetClusterPlacementGreedy(greedy C.bool) {
	p := "waterfill"
	if greedy {
		p = "greedy"
	}
	updateClusterConfig(api.ClusterConfigRequest{Placement: &p}, func(c *api.ClusterConfig) { c.Placement = p })
}

// ClusterShareItems returns the share switches as lines of "name\ttitle\ton"
// (see server.ClusterShareItems), as a C string the caller frees.
//
//export ClusterShareItems
func ClusterShareItems() *C.char {
	clusterMenu.Lock() // just refreshed by ClusterMenuState
	cfg := clusterMenu.cfg
	clusterMenu.Unlock()
	var sb strings.Builder
	for _, it := range server.ClusterShareItems(cfg) {
		on := "0"
		if it.On {
			on = "1"
		}
		sb.WriteString(it.Name + "\t" + it.Title + "\t" + on + "\n")
	}
	return C.CString(sb.String())
}
