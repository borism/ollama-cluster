//go:build !windows

package llm

import "golang.org/x/sys/unix"

// diskFree returns the bytes available to this user on path's filesystem,
// and the filesystem's total size.
func diskFree(path string) (free, total uint64, err error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, 0, err
	}
	return st.Bavail * uint64(st.Bsize), st.Blocks * uint64(st.Bsize), nil
}
