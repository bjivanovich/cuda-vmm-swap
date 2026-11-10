# CUDA VMM Hot-Swap Studio for Paseo

Connect your Paseo desktop and mobile workspace directly to a high-speed CUDA Virtual Memory Management (VMM) router. This plugin provides physical VRAM telemetry, instant sub-2.3s DMA model hot-swapping, and multi-agent model slot configuration.

## Features

- **Live VRAM Telemetry**: Displays real-time physical GPU memory consumption polled via low-level daemon RPCs.
- **Dynamic N-Agent Support**: Lists all configured agent model slots (such as Architect, Coder, Debugger, or any custom role like Tester or Reviewer) with 1-touch hot-swap buttons.
- **In-App Agent Configuration**: Register or customize model slots directly from Paseo without manually editing JSON files. Set the role identifier, local GGUF model path, context window, and optional modular PLE table.
- **Swap Mode Control**: Toggle between dynamic multi-agent pipeline swapping and single-model persistence.
- **Strict VRAM Eviction**: Trigger complete memory unmapping in under 60 milliseconds to ensure clean GPU headroom before loading large models.

## Setup

1. Run the local VMM router on the same machine (`python router/vmm_router.py` or via `master_harness.py`).
2. The plugin connects to `http://127.0.0.1:8000` to query status and dispatch hot-swap requests.
