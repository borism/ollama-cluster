import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Switch } from "@/components/ui/switch";
import { Text } from "@/components/ui/text";
import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  ServerStackIcon,
  ShareIcon,
  GlobeAltIcon,
  ExclamationTriangleIcon,
  ArrowsPointingOutIcon,
  CircleStackIcon,
} from "@heroicons/react/20/solid";
import {
  getClusterSettings,
  updateClusterSettings,
  getClusterPeers,
  getClusterModelSpillover,
  type ClusterSettingsData,
  type ClusterPeer,
  type ClusterPlacement,
} from "@/api";
import { isWindowsPlatform } from "@/lib/platform";

const clusterEnvVars = {
  enabled: "OLLAMA_CLUSTER",
  share: "OLLAMA_CLUSTER_SHARE",
  placement: "OLLAMA_CLUSTER_PLACEMENT",
  seeds: "OLLAMA_CLUSTER_SEEDS",
  cache_gb: "OLLAMA_CLUSTER_CACHE_GB",
} as const;

function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return `${gb.toFixed(1)} GB`;
}

function formatCacheSize(bytes: number): string {
  return bytes < 1024 ** 3
    ? `${Math.round(bytes / 1024 ** 2)} MB`
    : formatBytes(bytes);
}

function formatLastSeen(iso: string): string {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - new Date(iso).getTime()) / 1000),
  );
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ago`;
}

function PeerRow({ peer }: { peer: ClusterPeer }) {
  return (
    <li className="rounded-lg border border-neutral-200 dark:border-neutral-700 p-3 text-sm">
      <div className="flex items-center justify-between">
        <span className="font-medium text-black dark:text-neutral-100">
          {peer.addr}
        </span>
        <span className="text-neutral-500 dark:text-neutral-400">
          {peer.sharing ? "sharing" : "not sharing"}
        </span>
      </div>
      {peer.devices.map((device) => (
        <div
          key={device.name}
          className="mt-1 text-neutral-500 dark:text-neutral-400"
        >
          {device.name}: {formatBytes(device.free_memory)} free /{" "}
          {formatBytes(device.total_memory)}
        </div>
      ))}
      <div className="mt-1 text-neutral-500 dark:text-neutral-400">
        load {Math.round(peer.load * 100)}%
        {peer.latency_ms ? ` · ${peer.latency_ms} ms` : ""} · last seen{" "}
        {formatLastSeen(peer.last_seen)}
      </div>
    </li>
  );
}

// ClusterSettings renders the cluster-mode section of Settings.tsx: the
// on/off switch, the GPU-sharing switch, the placement policy, the
// tensor cache size, the cross-subnet seeds field, and a
// live status block of peers and any model spillover onto them. See
// docs/cluster.mdx for what these map to (OLLAMA_CLUSTER*).
export default function ClusterSettings() {
  const queryClient = useQueryClient();
  const [seedsInput, setSeedsInput] = useState("");
  const [cacheInput, setCacheInput] = useState("");
  // Errors are shown under the control that caused them.
  const [error, setError] = useState<{
    field: keyof ClusterSettingsData;
    message: string;
  } | null>(null);

  const { data: settings } = useQuery({
    queryKey: ["clusterSettings"],
    queryFn: getClusterSettings,
    // the tensor cache's size changes as peers load models
    refetchInterval: (query) => (query.state.data?.enabled ? 5000 : false),
  });

  const savedSeeds = settings?.seeds;
  useEffect(() => {
    if (savedSeeds !== undefined) setSeedsInput(savedSeeds);
  }, [savedSeeds]);

  const savedCacheGB = settings?.cache_gb;
  useEffect(() => {
    if (savedCacheGB !== undefined) setCacheInput(String(savedCacheGB));
  }, [savedCacheGB]);

  const enabled = settings?.enabled ?? false;

  const { data: peersResponse } = useQuery({
    queryKey: ["clusterPeers"],
    queryFn: getClusterPeers,
    enabled,
    refetchInterval: enabled ? 5000 : false,
  });

  const { data: spillover } = useQuery({
    queryKey: ["clusterModelSpillover"],
    queryFn: getClusterModelSpillover,
    enabled,
    refetchInterval: enabled ? 5000 : false,
  });

  const updateMutation = useMutation({
    mutationFn: updateClusterSettings,
    onSuccess: (saved) => {
      queryClient.setQueryData(["clusterSettings"], saved);
      setError(null);
    },
  });

  const save = (field: keyof ClusterSettingsData, next: ClusterSettingsData) =>
    updateMutation.mutate(next, {
      onError: (err: Error) =>
        setError({
          field,
          message: err.message.charAt(0).toUpperCase() + err.message.slice(1),
        }),
    });

  const handleToggle = (field: "enabled" | "share", value: boolean) => {
    if (!settings) return;
    save(field, { ...settings, [field]: value });
  };

  const commitSeeds = () => {
    if (!settings) return;
    const trimmed = seedsInput.trim();
    if (trimmed === settings.seeds) return;
    save("seeds", { ...settings, seeds: trimmed });
  };

  const commitCache = () => {
    if (!settings) return;
    const gb = Number(cacheInput);
    if (cacheInput.trim() === "" || !Number.isInteger(gb) || gb < 0) {
      setError({
        field: "cache_gb",
        message: "Enter a whole number of GB, or 0 to turn the cache off.",
      });
      setCacheInput(String(settings.cache_gb));
      return;
    }
    if (gb === settings.cache_gb) return;
    save("cache_gb", { ...settings, cache_gb: gb });
  };

  const errorFor = (field: keyof ClusterSettingsData) =>
    error?.field === field && (
      <p role="alert" className="mt-2 text-xs text-red-600 dark:text-red-400">
        {error.message}
      </p>
    );

  // A setting an OLLAMA_CLUSTER* variable decides can't be changed here.
  const locked = (field: keyof ClusterSettingsData) =>
    settings?.sources?.[field] === "env";
  const lockedNote = (field: keyof typeof clusterEnvVars) =>
    locked(field) && (
      <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
        Set by the {clusterEnvVars[field]} environment variable.
      </p>
    );

  if (!settings) return null;

  const peers = peersResponse?.peers ?? [];
  const spilledModels = (Array.isArray(spillover) ? spillover : []).filter(
    (model) => model.cluster_peers && model.cluster_peers.length > 0,
  );

  return (
    <>
      <Field>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start space-x-3 flex-1">
            <ServerStackIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div>
              <Label>Cluster mode</Label>
              <Description>
                Share GPU capacity with other computers running ollama-cluster
                on this network, and use theirs when a model doesn't fit
                locally.
              </Description>
            </div>
          </div>
          <div className="flex-shrink-0">
            <Switch
              checked={enabled}
              disabled={locked("enabled")}
              onChange={(checked) => handleToggle("enabled", checked)}
            />
          </div>
        </div>
        <div className="ml-8">
          {lockedNote("enabled")}
          {errorFor("enabled")}
        </div>
      </Field>

      <div className="flex items-start space-x-2 rounded-lg border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-900/20">
        <ExclamationTriangleIcon className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-600 dark:text-amber-400" />
        <Text className="text-sm text-amber-700 dark:text-amber-400">
          Only turn this on on a trusted network. Other computers on it can use
          this computer's GPU, and the connection between machines is
          unauthenticated.
        </Text>
      </div>

      <Field>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start space-x-3 flex-1">
            <ShareIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div>
              <Label>
                {settings.gpu
                  ? `Share the ${settings.gpu}`
                  : "Share this computer"}
              </Label>
              <Description>
                Let other computers in the cluster use this computer's spare GPU
                capacity. Turn off to use their capacity without donating your
                own.
              </Description>
            </div>
          </div>
          <div className="flex-shrink-0">
            <Switch
              checked={settings.share}
              disabled={!enabled || locked("share")}
              onChange={(checked) => handleToggle("share", checked)}
            />
          </div>
        </div>
        <div className="ml-8">
          {lockedNote("share")}
          {errorFor("share")}
        </div>
      </Field>

      <Field>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start space-x-3 flex-1">
            <ArrowsPointingOutIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div>
              <Label>Placement</Label>
              <Description>
                How peers are picked when a model doesn't fit. Waterfill weighs
                them by load and latency and can spread a model across several;
                greedy uses the peers with the most free memory, as few as
                possible.
              </Description>
            </div>
          </div>
          <div className="flex-shrink-0">
            <select
              aria-label="Placement"
              value={settings.placement}
              disabled={!enabled || locked("placement")}
              onChange={(e) =>
                save("placement", {
                  ...settings,
                  placement: e.target.value as ClusterPlacement,
                })
              }
              className="h-6 rounded-lg border border-zinc-950/10 bg-white px-2 py-0 text-sm text-black disabled:opacity-50 dark:border-white/10 dark:bg-neutral-800 dark:text-neutral-100"
            >
              <option value="waterfill">Waterfill</option>
              <option value="greedy">Greedy</option>
            </select>
          </div>
        </div>
        <div className="ml-8">
          {lockedNote("placement")}
          {errorFor("placement")}
        </div>
      </Field>

      <Field>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start space-x-3 flex-1">
            <CircleStackIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div>
              <Label>Tensor cache</Label>
              <Description>
                Up to this many GB of model weights this computer keeps from
                peers that used its GPU, so loading the same model again skips
                the network. The oldest files go first. 0 turns the cache off.
              </Description>
            </div>
          </div>
          <div className="flex flex-shrink-0 flex-col items-end gap-1">
            <div className="flex items-center gap-2">
              <input
                type="number"
                aria-label="Tensor cache size in GB"
                min={0}
                step={1}
                value={cacheInput}
                disabled={!enabled || !settings.share || locked("cache_gb")}
                onChange={(e) => setCacheInput(e.target.value)}
                onBlur={commitCache}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.currentTarget.blur();
                  }
                }}
                className="w-16 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none h-6 rounded-lg border border-zinc-950/10 bg-white px-2 py-0 text-sm text-black disabled:opacity-50 dark:border-white/10 dark:bg-neutral-800 dark:text-neutral-100"
              />
              <span className="text-sm text-neutral-500 dark:text-neutral-400">
                GB
              </span>
            </div>
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              {`${formatCacheSize(settings.cache_used_bytes ?? 0)} used`}
            </span>
          </div>
        </div>
        <div className="ml-8">
          {lockedNote("cache_gb")}
          {errorFor("cache_gb")}
        </div>
      </Field>

      <Field>
        <div className="flex items-start space-x-3">
          <GlobeAltIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          <div className="w-full">
            <Label>Peers on other subnets</Label>
            <Description>
              Comma-separated host:port list of peers to reach directly, for
              computers outside this one's broadcast domain. Leave empty on a
              flat LAN.
            </Description>
            <div className="mt-2">
              <Input
                value={seedsInput}
                disabled={!enabled || locked("seeds")}
                placeholder="192.0.2.10:11435, 192.0.2.11:11435"
                onChange={(e) => setSeedsInput(e.target.value)}
                onBlur={commitSeeds}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.currentTarget.blur();
                  }
                }}
              />
            </div>
            {lockedNote("seeds")}
            {errorFor("seeds")}
          </div>
        </div>
      </Field>

      {enabled && (
        <Field>
          <Label>Peers</Label>
          {!peersResponse?.enabled ? (
            <Description>Starting...</Description>
          ) : peers.length === 0 ? (
            <Description>
              Looking for other computers running ollama-cluster...
              {!isWindowsPlatform() &&
                " If none appear, check that Ollama is allowed under System Settings > Privacy & Security > Local Network."}
            </Description>
          ) : (
            <ul className="mt-2 space-y-2">
              {peers.map((peer) => (
                <PeerRow key={peer.id} peer={peer} />
              ))}
            </ul>
          )}
        </Field>
      )}

      {enabled && spilledModels.length > 0 && (
        <Field>
          <Label>Spilled onto peers</Label>
          <ul className="mt-2 space-y-1 text-sm text-neutral-500 dark:text-neutral-400">
            {spilledModels.map((model) => (
              <li key={model.name}>
                <span className="font-medium text-black dark:text-neutral-100">
                  {model.name}
                </span>
                :{" "}
                {model.cluster_peers
                  ?.map((peer) => `${peer.addr} (${formatBytes(peer.size)})`)
                  .join(", ")}
              </li>
            ))}
          </ul>
        </Field>
      )}
    </>
  );
}
