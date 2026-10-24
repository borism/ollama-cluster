package llm

import (
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeCacheFile writes data under the name ggml-rpc-server would give
// full (its FNV-1a hash), so passing a prefix of full as data makes the
// truncated file a failed write leaves behind.
func writeCacheFile(t *testing.T, dir string, full, data []byte, written time.Time) string {
	t.Helper()
	h := fnv.New64a()
	h.Write(full)
	path := filepath.Join(dir, fmt.Sprintf("%016x", h.Sum64()))
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, written, written); err != nil {
		t.Fatal(err)
	}
	return path
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func plentyFree(string) (uint64, uint64, error) { return 1 << 40, 1 << 41, nil }

func TestTidyRPCCacheRemovesTruncatedFiles(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	old := now.Add(-time.Hour)
	tensor := []byte("a weight tensor's bytes")
	good := writeCacheFile(t, dir, tensor, tensor, old)
	truncated := writeCacheFile(t, dir, []byte("another tensor's bytes"), []byte("another"), old)
	fresh := writeCacheFile(t, dir, []byte("still being written"), []byte("still"), now)

	if !tidyRPCCache(dir, 1<<30, now, plentyFree) {
		t.Fatal("tidyRPCCache reported no room with plenty free")
	}
	if !exists(good) || exists(truncated) {
		t.Errorf("good kept = %v (want true), truncated kept = %v (want false)", exists(good), exists(truncated))
	}
	if !exists(fresh) {
		t.Error("a file younger than rpcCacheSettle was checked, but it may still be mid-write")
	}

	// A later pass checks it, and only it: the good file isn't re-read.
	tidyRPCCache(dir, 1<<30, now.Add(2*rpcCacheSettle), plentyFree)
	if exists(fresh) {
		t.Error("truncated file kept after it settled")
	}
}

func TestTidyRPCCacheCapsSizeOldestFirst(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	tensor := func(s string) []byte { return []byte(fmt.Sprintf("%-100s", s)) }
	oldest := writeCacheFile(t, dir, tensor("a"), tensor("a"), now.Add(-3*time.Hour))
	middle := writeCacheFile(t, dir, tensor("b"), tensor("b"), now.Add(-2*time.Hour))
	newest := writeCacheFile(t, dir, tensor("c"), tensor("c"), now.Add(-1*time.Hour))

	tidyRPCCache(dir, 200, now, plentyFree)
	if exists(oldest) || !exists(middle) || !exists(newest) {
		t.Errorf("kept oldest=%v middle=%v newest=%v, want false true true", exists(oldest), exists(middle), exists(newest))
	}
}

func TestTidyRPCCacheFreesDiskSpace(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	tensor := []byte("a weight tensor's bytes")
	path := writeCacheFile(t, dir, tensor, tensor, now.Add(-time.Hour))

	full := func(string) (uint64, uint64, error) { return 1 << 20, 1 << 40, nil }
	if tidyRPCCache(dir, 1<<30, now, full) {
		t.Error("tidyRPCCache reported room to cache on a full disk")
	}
	if exists(path) {
		t.Error("cache file kept on a full disk")
	}
}

func TestRPCCacheBytes(t *testing.T) {
	models := t.TempDir()
	if got := RPCCacheBytes(models); got != 0 {
		t.Errorf("no cache dir: %d bytes, want 0", got)
	}
	dir := filepath.Join(models, "rpc")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	writeCacheFile(t, dir, []byte("one"), []byte("12345"), now)
	writeCacheFile(t, dir, []byte("two"), []byte("123"), now)
	os.WriteFile(filepath.Join(dir, rpcCacheCheckedMarker), []byte("not a tensor"), 0o644)
	if got := RPCCacheBytes(models); got != 8 {
		t.Errorf("RPCCacheBytes = %d, want 8 (tensors only, not the marker)", got)
	}
}

// diskWith fakes a volume of the given size whose free space is its size
// minus what's in dir, so deleting a cache file frees space.
func diskWith(dir string, size uint64) func(string) (uint64, uint64, error) {
	return func(string) (uint64, uint64, error) {
		used := RPCCacheBytes(filepath.Dir(dir))
		return size - used, size, nil
	}
}

// With no cap (rpcCacheUnlimited) only the free-space floor evicts, oldest
// first, and the floor is 5% of the volume clamped to 10-15 GiB.
func TestTidyRPCCacheUnlimitedEvictsOnlyUnderFloor(t *testing.T) {
	models := t.TempDir()
	dir := filepath.Join(models, "rpc")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	tensor := func(s string) []byte { return []byte(fmt.Sprintf("%-100s", s)) }
	oldest := writeCacheFile(t, dir, tensor("a"), tensor("a"), now.Add(-3*time.Hour))
	middle := writeCacheFile(t, dir, tensor("b"), tensor("b"), now.Add(-2*time.Hour))
	newest := writeCacheFile(t, dir, tensor("c"), tensor("c"), now.Add(-1*time.Hour))

	// A small volume has the 10 GiB floor, and with the cache's 300 bytes
	// 50 bytes are to spare: nothing to evict, however big the cache is.
	if !tidyRPCCache(dir, rpcCacheUnlimited, now, diskWith(dir, 10<<30+350)) || !exists(oldest) || !exists(middle) || !exists(newest) {
		t.Fatalf("evicted above the floor: kept %v %v %v", exists(oldest), exists(middle), exists(newest))
	}
	// With the cache full that is 50 bytes short of the floor: the oldest
	// file frees 100, which is enough.
	if !tidyRPCCache(dir, rpcCacheUnlimited, now, diskWith(dir, 10<<30+250)) {
		t.Error("reported no room after freeing enough")
	}
	if exists(oldest) || !exists(middle) || !exists(newest) {
		t.Errorf("kept oldest=%v middle=%v newest=%v, want false true true", exists(oldest), exists(middle), exists(newest))
	}

	// On a 1 TiB volume 5% is ~51 GiB, but the floor stops at 15 GiB: with
	// the disk reading 12 GiB free (over 10, under 15), everything goes.
	big := func(string) (uint64, uint64, error) { return 12 << 30, 1 << 40, nil }
	if tidyRPCCache(dir, rpcCacheUnlimited, now, big) {
		t.Error("reported room with 12 GiB free of 1 TiB")
	}
	if exists(middle) || exists(newest) {
		t.Error("files kept under the 15 GiB floor")
	}
}

// A big disk that is fairly full keeps its cache: 18 GiB free of 926 GiB is
// over the 15 GiB cap on the floor, though far under 5% of the volume.
func TestTidyRPCCacheLargeVolumeLimitedFree(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	tensor := []byte("a weight tensor's bytes")
	path := writeCacheFile(t, dir, tensor, tensor, now.Add(-time.Hour))
	disk := func(string) (uint64, uint64, error) { return 18 << 30, 926 << 30, nil }
	if !tidyRPCCache(dir, rpcCacheUnlimited, now, disk) || !exists(path) {
		t.Errorf("cache dropped with 18 GiB free of 926 GiB: kept=%v", exists(path))
	}
}

// An explicit cap still applies when there's plenty of space.
func TestTidyRPCCacheExplicitCapStillEnforced(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	tensor := func(s string) []byte { return []byte(fmt.Sprintf("%-100s", s)) }
	oldest := writeCacheFile(t, dir, tensor("a"), tensor("a"), now.Add(-2*time.Hour))
	newest := writeCacheFile(t, dir, tensor("b"), tensor("b"), now.Add(-1*time.Hour))
	tidyRPCCache(dir, 100, now, plentyFree)
	if exists(oldest) || !exists(newest) {
		t.Errorf("kept oldest=%v newest=%v, want false true", exists(oldest), exists(newest))
	}
}

// If free space can't be read, an unlimited cache falls back to the old
// 32 GB cap rather than growing unchecked; an explicit cap is unchanged.
func TestTidyRPCCacheUnknownFreeSpace(t *testing.T) {
	broken := func(string) (uint64, uint64, error) { return 0, 0, errors.New("statfs failed") }
	now := time.Now()
	// Two sparse 20 GB files: over 32 GB together, each under it. The
	// marker is newer than both so their (fake) contents aren't hash-checked.
	write := func() (dir, old, recent string) {
		dir = t.TempDir()
		old, recent = filepath.Join(dir, "00000000000000aa"), filepath.Join(dir, "00000000000000bb")
		for path, age := range map[string]time.Duration{old: 2 * time.Hour, recent: time.Hour} {
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Truncate(20 << 30); err != nil {
				t.Skip("no sparse files here:", err)
			}
			f.Close()
			os.Chtimes(path, now.Add(-age), now.Add(-age))
		}
		marker := filepath.Join(dir, rpcCacheCheckedMarker)
		os.WriteFile(marker, nil, 0o644)
		os.Chtimes(marker, now, now)
		return
	}

	dir, old, recent := write()
	if !tidyRPCCache(dir, rpcCacheUnlimited, now, broken) {
		t.Error("unknown free space should not count as no room")
	}
	if exists(old) || !exists(recent) {
		t.Errorf("unlimited with unknown free space: kept old=%v recent=%v, want false true (32 GB fallback)", exists(old), exists(recent))
	}

	dir, old, recent = write()
	tidyRPCCache(dir, 100<<30, now, broken)
	if !exists(old) || !exists(recent) {
		t.Errorf("explicit 100 GB cap with unknown free space evicted: old=%v recent=%v", exists(old), exists(recent))
	}
}
