package llm

import (
	"bytes"
	"log/slog"
	"net"
	"os"
	"os/exec"
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

	w.mu.Lock()
	cmd, done := w.cmd, w.done
	w.mu.Unlock()
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("kill worker process: %v", err)
	}
	select {
	case <-done:
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

func TestRPCRestartBackoff(t *testing.T) {
	var d time.Duration
	var got []time.Duration
	for range 8 {
		d = rpcRestartBackoff(d, time.Second)
		got = append(got, d)
	}
	want := []time.Duration{1, 2, 4, 8, 16, 32, 60, 60}
	for i := range want {
		if got[i] != want[i]*time.Second {
			t.Fatalf("backoff sequence = %v, want 1s,2s,4s,8s,16s,32s,1m,1m", got)
		}
	}
	if d := rpcRestartBackoff(time.Minute, 2*time.Minute); d != time.Second {
		t.Fatalf("backoff after a long healthy run = %v, want 1s", d)
	}
}

// TestRPCFakeServerHelper is not a test: it is the fake ggml-rpc-server the
// restart test re-executes this test binary as. It listens on
// $RPC_FAKE_PORT until killed.
func TestRPCFakeServerHelper(t *testing.T) {
	port := os.Getenv("RPC_FAKE_PORT")
	if port == "" {
		t.Skip("helper process only")
	}
	l, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		os.Exit(1)
	}
	for {
		c, err := l.Accept()
		if err != nil {
			os.Exit(1)
		}
		c.Close()
	}
}

// TestRPCWorkerRestartsOnCrash kills the worker process behind Stop's back
// and expects it back on the same port; after Stop it must stay down.
func TestRPCWorkerRestartsOnCrash(t *testing.T) {
	port, err := pickFreeTCPPort()
	if err != nil {
		t.Fatal(err)
	}
	w := &RPCWorker{
		exe:    os.Args[0],
		args:   []string{"-test.run=^TestRPCFakeServerHelper$"},
		env:    append(os.Environ(), "RPC_FAKE_PORT="+strconv.Itoa(port)),
		host:   "127.0.0.1",
		port:   port,
		stopCh: make(chan struct{}),
	}
	if err := w.spawn(); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	defer w.Stop()
	if err := w.waitUntilListening(5 * time.Second); err != nil {
		t.Fatal(err)
	}

	w.mu.Lock()
	first := w.cmd
	w.mu.Unlock()
	first.Process.Kill()

	deadline := time.Now().Add(10 * time.Second)
	var second *exec.Cmd
	for second == nil {
		w.mu.Lock()
		if w.cmd != first {
			second = w.cmd
		}
		w.mu.Unlock()
		if time.Now().After(deadline) {
			t.Fatal("worker was not respawned")
		}
		time.Sleep(50 * time.Millisecond)
	}
	for w.Port() != port {
		if time.Now().After(deadline) {
			t.Fatalf("Port() = %d, want %d back after a crash", w.Port(), port)
		}
		time.Sleep(50 * time.Millisecond)
	}

	w.Stop()
	time.Sleep(2 * time.Second) // longer than the min backoff
	if w.Port() != 0 {
		t.Fatalf("Port() = %d after Stop, want 0 (no restart)", w.Port())
	}
	w.mu.Lock()
	third := w.cmd
	w.mu.Unlock()
	if third != second {
		t.Fatal("worker was respawned after Stop")
	}
}
