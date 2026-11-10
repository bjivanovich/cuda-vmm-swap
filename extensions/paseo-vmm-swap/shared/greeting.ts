import { defineRpc } from "@getpaseo/plugin";
import { z } from "zod";

export const vmmModelSchema = z.object({
  id: z.string(),
  alias: z.string(),
  role: z.string(),
  model_path: z.string(),
  context: z.number(),
  ctk: z.string().optional(),
  ctv: z.string().optional(),
  is_active: z.boolean(),
});

export const vmmStatusRpc = defineRpc({
  name: "vmm.status",
  input: z.object({}),
  output: z.object({
    active_vram_model: z.string().nullable(),
    staged_ram_model: z.string().nullable(),
    physical_vram_used_mb: z.number(),
    last_swap_latency_seconds: z.number(),
    swap_enabled: z.boolean(),
    configured_models: z.array(z.string()),
    models: z.array(vmmModelSchema),
  }),
});

export const vmmSwapRpc = defineRpc({
  name: "vmm.swap",
  input: z.object({ model: z.string() }),
  output: z.object({
    status: z.string(),
    active_model: z.string().nullable(),
    swap_latency_seconds: z.number(),
  }),
});

export const vmmToggleSwapRpc = defineRpc({
  name: "vmm.toggle_swap",
  input: z.object({}),
  output: z.object({
    swap_enabled: z.boolean(),
  }),
});

export const vmmEvictRpc = defineRpc({
  name: "vmm.evict_all",
  input: z.object({}),
  output: z.object({
    status: z.string(),
    vram_used_mb: z.number(),
  }),
});

export const vmmConfigureModelRpc = defineRpc({
  name: "vmm.configure_model",
  input: z.object({
    id: z.string(),
    model_path: z.string(),
    alias: z.string().optional(),
    role: z.string().optional(),
    context: z.number().optional(),
    ple_path: z.string().optional(),
  }),
  output: z.object({
    status: z.string(),
    model_id: z.string(),
    configured_models: z.array(z.string()),
  }),
});
