package llm

import "golang.org/x/sys/windows"

// diskFree returns the bytes available to this user on path's volume, and
// the volume's total size.
func diskFree(path string) (free, total uint64, err error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, 0, err
	}
	if err := windows.GetDiskFreeSpaceEx(p, &free, &total, nil); err != nil {
		return 0, 0, err
	}
	return free, total, nil
}
