// Maps the cluster settings (share + share_devices, see api.ClusterConfig)
// to one on/off switch per local GPU and back.

export interface ShareDevice {
  name: string; // ggml device name ("CUDA1"), what share_devices lists
}

export interface ShareState {
  share: boolean;
  share_devices: string;
}

const listed = (shareDevices: string) =>
  shareDevices
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

// Names of the devices currently shared: none when sharing is off, all
// when share_devices is empty.
export function sharedDeviceNames(
  state: ShareState,
  devices: ShareDevice[],
): string[] {
  if (!state.share) return [];
  const want = listed(state.share_devices);
  return devices
    .map((d) => d.name)
    .filter((n) => want.length === 0 || want.includes(n.toLowerCase()));
}

// The settings to save once `on` are the devices to share. All on is
// "share everything" (empty list), so a GPU added later is shared too.
export function shareStateFor(
  on: string[],
  devices: ShareDevice[],
): ShareState {
  if (on.length === 0) return { share: false, share_devices: "" };
  const names = devices.map((d) => d.name).filter((n) => on.includes(n));
  return {
    share: true,
    share_devices: names.length === devices.length ? "" : names.join(","),
  };
}
