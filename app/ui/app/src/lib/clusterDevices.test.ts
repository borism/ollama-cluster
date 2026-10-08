import { describe, expect, it } from "vitest";
import { shareStateFor, sharedDeviceNames } from "./clusterDevices";

const devs = [{ name: "CUDA0" }, { name: "CUDA1" }];

describe("clusterDevices", () => {
  it("reads the settings", () => {
    expect(
      sharedDeviceNames({ share: false, share_devices: "" }, devs),
    ).toEqual([]);
    expect(sharedDeviceNames({ share: true, share_devices: "" }, devs)).toEqual(
      ["CUDA0", "CUDA1"],
    );
    expect(
      sharedDeviceNames({ share: true, share_devices: "cuda1" }, devs),
    ).toEqual(["CUDA1"]);
  });

  it("writes the settings", () => {
    expect(shareStateFor(["CUDA0", "CUDA1"], devs)).toEqual({
      share: true,
      share_devices: "",
    });
    expect(shareStateFor([], devs)).toEqual({
      share: false,
      share_devices: "",
    });
    expect(shareStateFor(["CUDA1"], devs)).toEqual({
      share: true,
      share_devices: "CUDA1",
    });
  });
});
