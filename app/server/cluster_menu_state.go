// No build tag: builds and tests on Linux too.

package server

import (
	"sync"

	"github.com/borism/ollama-cluster/api"
)

// ClusterMenuCache holds the cluster settings the menu-bar menu shows, so
// the menu can draw them without waiting for the server.
type ClusterMenuCache struct {
	sync.Mutex
	Cfg     api.ClusterConfig
	Known   bool // Cfg came from the server at least once
	Pending int  // changes still being applied; Cfg shows them already
}

// Apply records the server's answer to GET /api/cluster/config (err is
// the request's error). A failed or slow request keeps the last settings
// instead of blanking them, and an answer that may predate a change still
// being applied is dropped. The caller holds the lock.
func (c *ClusterMenuCache) Apply(fresh *api.ClusterConfig, err error) {
	if err != nil || c.Pending > 0 {
		return
	}
	c.Cfg, c.Known = *fresh, true
}
