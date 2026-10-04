import {
  QueryClient,
  defaultScheduler,
  notifyManager,
} from "@tanstack/react-query";
import type { ChangeEvent } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ClusterListResponse, ClusterSettingsData } from "@/api";
import { Switch } from "./ui/switch";
import { Input } from "./ui/input";
import ClusterSettings from "./ClusterSettings";

const mocks = vi.hoisted(() => ({
  getClusterSettings: vi.fn(),
  updateClusterSettings: vi.fn(),
  getClusterPeers: vi.fn(),
  getClusterModelSpillover: vi.fn(),
}));

vi.mock("@/api", () => ({
  getClusterSettings: mocks.getClusterSettings,
  updateClusterSettings: mocks.updateClusterSettings,
  getClusterPeers: mocks.getClusterPeers,
  getClusterModelSpillover: mocks.getClusterModelSpillover,
}));

let queryClient: QueryClient;

// Bind the real hooks to an explicit QueryClient instead of standing up a
// <QueryClientProvider>, matching CodexDesktopRow.test.tsx.
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return Object.assign({}, actual, {
    useQueryClient: () => queryClient,
    useQuery: (options: Parameters<typeof actual.useQuery>[0]) =>
      actual.useQuery(options, queryClient),
    useMutation: (options: Parameters<typeof actual.useMutation>[0]) =>
      actual.useMutation(options, queryClient),
  });
});

function settings(
  overrides: Partial<ClusterSettingsData> = {},
): ClusterSettingsData {
  return {
    enabled: false,
    share: true,
    share_devices: "",
    seeds: "",
    placement: "waterfill",
    cache_gb: 32,
    ...overrides,
  };
}

function peersResponse(
  overrides: Partial<ClusterListResponse> = {},
): ClusterListResponse {
  return { enabled: true, peers: [], ...overrides };
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  queryClient = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  mocks.getClusterSettings.mockReset().mockResolvedValue(settings());
  mocks.updateClusterSettings.mockReset();
  mocks.getClusterPeers.mockReset().mockResolvedValue(peersResponse());
  mocks.getClusterModelSpillover.mockReset().mockResolvedValue([]);
});

afterEach(() => {
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderCluster(): Promise<ReactTestRenderer> {
  let renderer!: ReactTestRenderer;
  await act(async () => {
    renderer = create(<ClusterSettings />);
  });
  await flush();
  return renderer;
}

function html(renderer: ReactTestRenderer): string {
  return JSON.stringify(renderer.toJSON());
}

describe("ClusterSettings", () => {
  it("always shows the trusted-network warning, even when off", async () => {
    const renderer = await renderCluster();
    expect(html(renderer)).toContain("trusted network");
    await act(async () => renderer.unmount());
  });

  it("shows the empty state, then renders peers once cluster mode is on", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    mocks.getClusterPeers.mockResolvedValue(
      peersResponse({
        peers: [
          {
            id: "9f3a1c7e2b804d61",
            addr: "192.0.2.11:11435",
            sharing: true,
            devices: [
              {
                name: "CUDA0",
                total_memory: 16_000_000_000,
                free_memory: 8_000_000_000,
              },
            ],
            load: 0.12,
            latency_ms: 4,
            last_seen: new Date().toISOString(),
          },
        ],
      }),
    );

    const renderer = await renderCluster();

    expect(html(renderer)).toContain("192.0.2.11:11435");
    expect(html(renderer)).toContain("sharing");
    await act(async () => renderer.unmount());
  });

  it("says it's looking for peers when none have been seen yet", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    const renderer = await renderCluster();

    expect(html(renderer)).toContain("Looking for other computers");
    expect(html(renderer)).toContain("Local Network");
    await act(async () => renderer.unmount());
  });

  it("calls the API when the cluster mode switch is toggled", async () => {
    mocks.updateClusterSettings.mockResolvedValue(settings({ enabled: true }));
    const renderer = await renderCluster();

    const clusterSwitch = renderer.root.findAllByType(Switch)[0];
    await act(async () => {
      clusterSwitch.props.onChange(true);
      await Promise.resolve();
    });

    expect(mocks.updateClusterSettings).toHaveBeenCalledWith(
      settings({ enabled: true }),
    );
    await act(async () => renderer.unmount());
  });

  it("shows one switch per GPU and saves the ones left on", async () => {
    const devices = [
      { name: "CUDA0", description: "NVIDIA RTX A4500", total_memory: 2 ** 34 },
      { name: "CUDA1", description: "NVIDIA RTX A2000", total_memory: 2 ** 33 },
    ];
    mocks.getClusterSettings.mockResolvedValue(
      settings({ enabled: true, devices }),
    );
    mocks.updateClusterSettings.mockResolvedValue(settings());
    const renderer = await renderCluster();

    expect(html(renderer)).toContain("NVIDIA RTX A2000");
    const switches = renderer.root.findAllByType(Switch);
    expect(switches.map((sw) => sw.props.checked)).toEqual([true, true, true]);
    await act(async () => {
      switches[1].props.onChange(false);
      await Promise.resolve();
    });

    expect(mocks.updateClusterSettings).toHaveBeenCalledWith(
      settings({ enabled: true, devices, share_devices: "CUDA1" }),
    );
    await act(async () => renderer.unmount());
  });

  it("labels the sharing switch CPU on a machine without a GPU", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings());
    const renderer = await renderCluster();

    expect(html(renderer)).toContain("Share this computer's CPU");
    await act(async () => renderer.unmount());
  });

  it("locks a setting an environment variable decides", async () => {
    mocks.getClusterSettings.mockResolvedValue(
      settings({ enabled: true, sources: { enabled: "env", share: "config" } }),
    );
    const renderer = await renderCluster();

    const [clusterSwitch, shareSwitch] = renderer.root.findAllByType(Switch);
    expect(clusterSwitch.props.disabled).toBe(true);
    expect(shareSwitch.props.disabled).toBe(false);
    expect(html(renderer)).toContain("OLLAMA_CLUSTER");
    expect(html(renderer)).not.toContain("OLLAMA_CLUSTER_SHARE");
    await act(async () => renderer.unmount());
  });

  it("saves the placement picked in the select", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    mocks.updateClusterSettings.mockResolvedValue(
      settings({ enabled: true, placement: "greedy" }),
    );
    const renderer = await renderCluster();

    await act(async () => {
      renderer.root
        .findByType("select")
        .props.onChange({ target: { value: "greedy" } });
      await Promise.resolve();
    });

    expect(mocks.updateClusterSettings).toHaveBeenCalledWith(
      settings({ enabled: true, placement: "greedy" }),
    );
    await act(async () => renderer.unmount());
  });

  it("saves the tensor cache size on blur and rejects a bad value", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    mocks.updateClusterSettings.mockResolvedValue(
      settings({ enabled: true, cache_gb: 64 }),
    );
    const renderer = await renderCluster();
    const input = () =>
      renderer.root.findByProps({ "aria-label": "Tensor cache size in GB" });
    const type = async (value: string) => {
      await act(async () => {
        input().props.onChange({ target: { value } });
      });
      await act(async () => {
        input().props.onBlur();
        await Promise.resolve();
      });
    };

    await type("1.5");
    expect(mocks.updateClusterSettings).not.toHaveBeenCalled();
    expect(html(renderer)).toContain("Enter a whole number of GB");
    expect(input().props.value).toBe("32");

    await type("64");
    expect(mocks.updateClusterSettings).toHaveBeenCalledWith(
      settings({ enabled: true, cache_gb: 64 }),
    );
    await act(async () => renderer.unmount());
  });

  it("shows how much the tensor cache holds", async () => {
    mocks.getClusterSettings.mockResolvedValue(
      settings({ enabled: true, cache_used_bytes: 3.5 * 1024 ** 3 }),
    );
    let renderer = await renderCluster();
    expect(html(renderer)).toContain("3.5 GB used");
    await act(async () => renderer.unmount());

    mocks.getClusterSettings.mockResolvedValue(
      settings({ enabled: true, cache_used_bytes: 300 * 1024 ** 2 }),
    );
    renderer = await renderCluster();
    expect(html(renderer)).toContain("300 MB used");
    await act(async () => renderer.unmount());
  });

  it("shows latency only when it has been measured", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    const peer = {
      id: "9f3a1c7e2b804d61",
      addr: "192.0.2.11",
      sharing: true,
      devices: [],
      load: 0,
      last_seen: new Date().toISOString(),
    };
    mocks.getClusterPeers.mockResolvedValue(
      peersResponse({
        peers: [peer, { ...peer, id: "0c88e4a1f5d23b90", latency_ms: 4 }],
      }),
    );
    const renderer = await renderCluster();

    expect(html(renderer)).toContain(" · 4 ms");
    expect(html(renderer)).not.toContain(" · -");
    await act(async () => renderer.unmount());
  });

  it("shows a failed toggle's error under that switch, not the seeds field", async () => {
    mocks.updateClusterSettings.mockRejectedValue(
      new Error("failed to save cluster settings"),
    );
    const renderer = await renderCluster();

    await act(async () => {
      renderer.root.findAllByType(Switch)[0].props.onChange(true);
      await Promise.resolve();
    });
    await flush();

    const alerts = renderer.root.findAll(
      (node) => node.type === "p" && node.props.role === "alert",
    );
    expect(alerts).toHaveLength(1);
    expect(html(renderer)).toContain("Failed to save cluster settings");
    // rendered inside the cluster-mode switch's own Field (via the
    // indented wrapper that lines it up with the label)
    expect(alerts[0].parent?.parent?.findAllByType(Switch)[0]).toBe(
      renderer.root.findAllByType(Switch)[0],
    );
    await act(async () => renderer.unmount());
  });

  it("shows an inline error when the seeds field fails validation", async () => {
    mocks.getClusterSettings.mockResolvedValue(settings({ enabled: true }));
    mocks.updateClusterSettings.mockRejectedValue(
      new Error('invalid seed "nope": expected host:port'),
    );
    const renderer = await renderCluster();

    const seedsInput = renderer.root.findByType(Input);
    await act(async () => {
      seedsInput.props.onChange({
        target: { value: "nope" },
      } as ChangeEvent<HTMLInputElement>);
    });
    await act(async () => {
      seedsInput.props.onBlur();
      await Promise.resolve();
    });

    expect(html(renderer)).toContain("Invalid seed");
    await act(async () => renderer.unmount());
  });
});
