import { defineRpc } from "@getpaseo/plugin";
import { z } from "zod";

export const vmmStatusRpc = defineRpc({
  name: "vmm.status",
  input: z.object({}),
  output: z.object({
    active_vram_model: z.string().nullable(),
    physical_vram_used_mb: z.number(),
    last_swap_latency_seconds: z.number(),
    swap_enabled: z.boolean(),
    configured_models: z.array(z.string()),
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
