import type { RpcInput } from "@getpaseo/plugin";
import { vmmStatusRpc, vmmSwapRpc } from "../shared/greeting";

const ROUTER_BASE = "http://127.0.0.1:8000";

export async function getVmmStatus(_input: RpcInput<typeof vmmStatusRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/status`);
    if (!res.ok) {
      throw new Error(`Router returned HTTP ${res.status}`);
    }
    const data = await res.json();
    return {
      active_vram_model: data.active_vram_model ?? null,
      physical_vram_used_mb: data.physical_vram_used_mb ?? 0,
      last_swap_latency_seconds: data.last_swap_latency_seconds ?? 0.0,
      swap_enabled: Boolean(data.swap_enabled),
      configured_models: data.configured_models ?? ["architect", "coder", "debugger"],
    };
  } catch (err: any) {
    return {
      active_vram_model: "offline",
      physical_vram_used_mb: 0,
      last_swap_latency_seconds: 0.0,
      swap_enabled: false,
      configured_models: [],
    };
  }
}

export async function swapVmmModel({ model }: RpcInput<typeof vmmSwapRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/v1/vmm/swap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const data = await res.json();
    return {
      status: data.status ?? "success",
      active_model: data.active_model ?? model,
      swap_latency_seconds: data.swap_latency_seconds ?? 0.0,
    };
  } catch (err: any) {
    return {
      status: `error: ${err.message}`,
      active_model: null,
      swap_latency_seconds: 0.0,
    };
  }
}
