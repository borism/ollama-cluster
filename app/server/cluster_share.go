// No build tag: builds and tests on Linux too.

package server

import (
	"strings"

	"github.com/borism/ollama-cluster/api"
)

// ShareItem is one menu-bar switch: a GPU (Name is what share_devices
// takes), or the whole computer when it has none (Name is "").
type ShareItem struct {
	Name  string
	Title string
	On    bool
}

func listedDevices(shareDevices string) map[string]bool {
	m := map[string]bool{}
	for _, s := range strings.Split(shareDevices, ",") {
		if s = strings.ToLower(strings.TrimSpace(s)); s != "" {
			m[s] = true
		}
	}
	return m
}

// ClusterShareItems lists the switches for cfg: one per GPU titled "Share
// <description>" (plus " (<name>)" if two share a description), "Share the
// <description> GPU" for a single GPU, "Share this computer's CPU" for none.
// Mirrors sharedDeviceNames in app/ui/app/src/lib/clusterDevices.ts.
func ClusterShareItems(cfg api.ClusterConfig) []ShareItem {
	devs := cfg.Devices
	switch len(devs) {
	case 0:
		return []ShareItem{{Title: "Share this computer's CPU", On: cfg.Share}}
	case 1:
		return []ShareItem{{Name: devs[0].Name, Title: "Share the " + ClusterGPUName([]string{devs[0].Description}), On: cfg.Share}}
	}
	want := listedDevices(cfg.ShareDevices)
	count := map[string]int{}
	for _, d := range devs {
		count[d.Description]++
	}
	items := make([]ShareItem, len(devs))
	for i, d := range devs {
		title := "Share " + d.Description
		if count[d.Description] > 1 {
			title += " (" + d.Name + ")"
		}
		items[i] = ShareItem{d.Name, title, cfg.Share && (len(want) == 0 || want[strings.ToLower(d.Name)])}
	}
	return items
}

// ClusterShareToggle returns the share and share_devices settings to save
// after flipping the switch for device name ("" for the CPU-only switch).
// Mirrors shareStateFor in clusterDevices.ts: all on is "share everything"
// (empty list), none on is share off.
func ClusterShareToggle(cfg api.ClusterConfig, name string) (share bool, shareDevices string) {
	if name == "" {
		return !cfg.Share, ""
	}
	var on []string
	for _, it := range ClusterShareItems(cfg) {
		if it.On != (it.Name == name) { // flip name, keep the rest
			on = append(on, it.Name)
		}
	}
	if len(on) == 0 {
		return false, ""
	}
	if len(on) == len(cfg.Devices) {
		return true, ""
	}
	return true, strings.Join(on, ",")
}
