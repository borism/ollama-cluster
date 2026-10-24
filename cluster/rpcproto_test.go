package cluster

import (
	"os"
	"strings"
	"testing"
)

// A sync PR that bumps LLAMA_CPP_VERSION must rerun `go generate ./cluster`,
// otherwise the advertised RPC protocol version is stale.
func TestRPCProtoMatchesPin(t *testing.T) {
	b, err := os.ReadFile("../LLAMA_CPP_VERSION")
	if err != nil {
		t.Fatal(err)
	}
	if pin := strings.TrimSpace(string(b)); pin != RPCLlamaCppTag {
		t.Fatalf("rpcproto_gen.go was generated from llama.cpp %s but LLAMA_CPP_VERSION is %s; run `go generate ./cluster`", RPCLlamaCppTag, pin)
	}
}
