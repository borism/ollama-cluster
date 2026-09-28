package llm

import (
	"bytes"
	"log/slog"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestRPCWorkerStartStop is a real spawn/stop round trip against the actual
// ggml-rpc-server binary. It mirrors TestFindLlamaServer's "may or may not
// be built" reality: skip cleanly when the binary isn't in lib/ollama/ (i.e.
// cmake --preset cpu / --build hasn't been run in this checkout).
func TestRPCWorkerStartStop(t *testing.T) {
	if _, err := FindRPCWorker(); err != nil {
		t.Skipf("ggml-rpc-server not found, skipping: %v", err)
	}

	cacheDir := t.TempDir()
	w, err := StartRPCWorker(0, cacheDir)
	if err != nil {
		t.Fatalf("StartRPCWorker: %v", err)
	}
	defer w.Stop()

	if w.Port() == 0 {
		t.Fatal("Port() = 0, want a real port")
	}
	addr := w.Addr()
	if addr == "" {
		t.Fatal("Addr() = \"\", want host:port")
	}
	if _, portStr, err := net.SplitHostPort(addr); err != nil || portStr != strconv.Itoa(w.Port()) {
		t.Fatalf("Addr() = %q inconsistent with Port() = %d", addr, w.Port())
	}

	// Dial it for real: the worker binds every interface, but a loopback
	// dial always reaches it.
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(w.Port())), 2*time.Second)
	if err != nil {
		t.Fatalf("dial running worker: %v", err)
	}
	conn.Close()

	if err := w.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if w.Port() != 0 || w.Addr() != "" {
		t.Fatalf("worker still reports running after Stop: port=%d addr=%q", w.Port(), w.Addr())
	}

	// cache dir was requested, so ggml-rpc-server should have created a
	// subdirectory under it (fs_get_cache_directory + "rpc/" in
	// tools/rpc/rpc-server.cpp).
	entries, err := os.ReadDir(cacheDir)
	if err != nil || len(entries) == 0 {
		t.Logf("cache dir %s has no entries after a connection with no RPC traffic (expected -- cache only fills on actual tensor transfer)", cacheDir)
	}
}

// TestRPCWorkerLogsUnexpectedExit covers the gap two real fleet crashes hit:
// the worker's own process dying leaves Port() correctly reporting 0 (so
// peers stop being told this instance shares), but until now nothing said
// *why* -- cmd.Wait() was checked only to close a channel. A killed-out-from-
// under-it worker should at least leave a log line behind.
func TestRPCWorkerLogsUnexpectedExit(t *testing.T) {
	if _, err := FindRPCWorker(); err != nil {
		t.Skipf("ggml-rpc-server not found, skipping: %v", err)
	}

	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(prev)

	w, err := StartRPCWorker(0, "")
	if err != nil {
		t.Fatalf("StartRPCWorker: %v", err)
	}
	defer w.Stop()

	if err := w.cmd.Process.Kill(); err != nil {
		t.Fatalf("kill worker process: %v", err)
	}
	select {
	case <-w.done:
	case <-time.After(2 * time.Second):
		t.Fatal("worker did not report exit after being killed")
	}

	if got := buf.String(); !strings.Contains(got, "rpc worker exited") {
		t.Fatalf("log output = %q, want a line about the worker exiting", got)
	}
}

// TestRPCWorkerWaitUntilListening is a pure unit test of the readiness loop:
// no real ggml-rpc-server needed, just a plain TCP listener standing in for
// one, and a pre-closed done channel standing in for one that died early.
func TestRPCWorkerWaitUntilListening(t *testing.T) {
	t.Run("succeeds once something is listening", func(t *testing.T) {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		defer l.Close()

		w := &RPCWorker{
			port:   l.Addr().(*net.TCPAddr).Port,
			done:   make(chan struct{}),
			status: NewStatusWriter(nil),
		}
		if err := w.waitUntilListening(2 * time.Second); err != nil {
			t.Fatalf("waitUntilListening: %v", err)
		}
	})

	t.Run("fails fast when the process already exited", func(t *testing.T) {
		done := make(chan struct{})
		close(done)
		status := NewStatusWriter(nil)
		status.AppendError("error: no devices found")

		w := &RPCWorker{
			port:   1, // nothing listens on port 1
			done:   done,
			status: status,
		}
		err := w.waitUntilListening(2 * time.Second)
		if err == nil {
			t.Fatal("waitUntilListening: expected an error, got nil")
		}
		if got := err.Error(); !strings.Contains(got, "no devices found") {
			t.Fatalf("waitUntilListening error = %q, want it to include the captured stderr", got)
		}
	})

	t.Run("times out when nothing ever listens", func(t *testing.T) {
		w := &RPCWorker{
			port:   1, // nothing listens on port 1
			done:   make(chan struct{}),
			status: NewStatusWriter(nil),
		}
		start := time.Now()
		err := w.waitUntilListening(150 * time.Millisecond)
		if err == nil {
			t.Fatal("waitUntilListening: expected a timeout error, got nil")
		}
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Fatalf("waitUntilListening took %v, want it to respect the timeout", elapsed)
		}
	})
}

func TestPickFreeTCPPort(t *testing.T) {
	port, err := pickFreeTCPPort()
	if err != nil {
		t.Fatalf("pickFreeTCPPort: %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Fatalf("pickFreeTCPPort = %d, want a valid port", port)
	}

	// The port should be immediately reusable.
	l, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		t.Fatalf("port %d not usable right after picking it: %v", port, err)
	}
	l.Close()
}
