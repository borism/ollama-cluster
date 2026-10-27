package server

import (
	"errors"
	"testing"

	"github.com/borism/ollama-cluster/api"
)

func TestClusterMenuCacheKeepsLastGood(t *testing.T) {
	var c ClusterMenuCache
	c.Apply(nil, errors.New("timeout"))
	if c.Known {
		t.Fatal("known before the server ever answered")
	}

	c.Apply(&api.ClusterConfig{Enabled: true, Share: true}, nil)
	c.Apply(nil, errors.New("timeout"))
	if !c.Known || !c.Cfg.Enabled || !c.Cfg.Share {
		t.Fatalf("a failed request blanked the last settings: %+v known=%v", c.Cfg, c.Known)
	}

	c.Pending = 1 // a change is still being applied
	c.Apply(&api.ClusterConfig{}, nil)
	if !c.Cfg.Enabled {
		t.Fatal("an answer that predates a pending change replaced it")
	}
}
