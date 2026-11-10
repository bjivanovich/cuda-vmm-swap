import type { RpcInput } from "@getpaseo/plugin";
import {
  vmmConfigureModelRpc,
  vmmEvictRpc,
  vmmStatusRpc,
  vmmSwapRpc,
  vmmToggleSwapRpc,
} from "../shared/greeting";

const ROUTER_BASE = "http://127.0.0.1:8000";

export async function getVmmStatus(_input: RpcInput<typeof vmmStatusRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/status`);
    if (!res.ok) {
      throw new Error(`Router returned HTTP ${res.status}`);
    }
    const data = await res.json();
    const rawModels = data.models ?? {};
    const activeModel = data.active_vram_model;

    const modelsList = Object.entries(rawModels).map(([id, info]: [string, any]) => ({
      id,
      alias: info.alias ?? id,
      role: info.role ?? `Agent (${id})`,
      model_path: info.model_path ?? "",
      context: Number(info.context ?? 98304),
      ctk: info.ctk ?? "q8_0",
      ctv: info.ctv ?? "q8_0",
      is_active: id === activeModel,
    }));

    return {
      active_vram_model: activeModel ?? null,
      staged_ram_model: data.staged_ram_model ?? null,
      physical_vram_used_mb: Number(data.physical_vram_used_mb ?? 0),
      last_swap_latency_seconds: Number(data.last_swap_latency_seconds ?? 0.0),
      swap_enabled: Boolean(data.swap_enabled),
      configured_models: data.configured_models ?? Object.keys(rawModels),
      models: modelsList,
    };
  } catch (_err) {
    return {
      active_vram_model: "offline",
      staged_ram_model: null,
      physical_vram_used_mb: 0,
      last_swap_latency_seconds: 0.0,
      swap_enabled: false,
      configured_models: [],
      models: [],
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
      swap_latency_seconds: Number(data.swap_latency_seconds ?? 0.0),
    };
  } catch (err: any) {
    return {
      status: `error: ${err.message}`,
      active_model: null,
      swap_latency_seconds: 0.0,
    };
  }
}

export async function toggleVmmSwap(_input: RpcInput<typeof vmmToggleSwapRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/api/toggle-swap`, { method: "POST" });
    const data = await res.json();
    return { swap_enabled: Boolean(data.swap_enabled) };
  } catch (_err) {
    return { swap_enabled: false };
  }
}

export async function evictVmmAll(_input: RpcInput<typeof vmmEvictRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/api/evict-all`, { method: "POST" });
    const data = await res.json();
    return {
      status: data.status ?? "evicted",
      vram_used_mb: Number(data.vram_used_mb ?? 0),
    };
  } catch (err: any) {
    return { status: `error: ${err.message}`, vram_used_mb: 0 };
  }
}

export async function configureVmmModel(input: RpcInput<typeof vmmConfigureModelRpc>) {
  try {
    const res = await fetch(`${ROUTER_BASE}/api/models/configure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
    const data = await res.json();
    return {
      status: data.status ?? "success",
      model_id: data.model_id ?? input.id,
      configured_models: data.configured_models ?? [],
    };
  } catch (err: any) {
    return {
      status: `error: ${err.message}`,
      model_id: input.id,
      configured_models: [],
    };
  }
}
