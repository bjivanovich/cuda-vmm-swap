# Changelog

All notable changes to the `cuda-vmm-swap` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.3.0] - 2026-11-08

### Added
- **Engine-Aware Memory Budget Calculator (`router/memory_advisor.py`)**:
  - Auto-detects physical GPU VRAM and Total System RAM.
  - Computes the Windows WDDM 50% pinned host RAM cap (`cudaHostAlloc`), preventing `cudaErrorMemoryAllocation` during sliding window pre-staging.
  - Reads GGUF binary headers directly to extract exact architecture tensor shapes: `block_count` (layers), `head_count_kv` (KV heads), and `head_dim`.
  - Factors exact bytes-per-element according to target engine:
    - `llamAmpere-Ultra`: TurboQuant (`turbo5`, `turbo4`, `turbo3`, `turbo2`), SJ-KVaRN fused rotation codec (`GGML_SJKVARN_FUSED_ROT`), Q8_0, Q4_0, FP16.
    - `llama.cpp` upstream: FP16, Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, IQ4_NL.
  - Dynamic viability evaluation (`fits_vram` and `fits_pinned_ram`).
- **Web Control Studio Dashboard (`router/dashboard.html`)**:
  - Integrated mobile touch-friendly web GUI served at `/` and `/ui`.
  - Real-time physical VRAM and System RAM gauges with the Windows WDDM 50% threshold marker.
  - Model slot cards with live viability badges and `<2.3s` manual swap trigger buttons.
  - Interactive prompt testing playground streaming completions directly from the active model in VRAM.
- **Dynamic Swap Mode & Multi-Slot Support**:
  - Global `swap_enabled` toggle: support for single-model execution (swap disabled) or dynamic multi-model swapping (1, 2, 3, or N models).
  - Environment variable resolution (`%LLAMA_SERVER_EXE%`, `%MODELS_DIR%`) and fallback binary discovery.
- **MCP Tool `vmm_get_memory_budget`**:
  - Added to `mcp/vmm_mcp_server.py`, exposing memory budget assessment and hardware limits to MCP clients (Paseo, Claude Desktop, Antigravity).
- **Harness Upgrades (`master_harness.py`)**:
  - Added `--swap [on|off]`, `--open-ui`, and `--check-memory` flags.
  - Auto-detection of Python interpreter (project-bundled or sys.executable).

### Changed
- `router/router_config.json`: Added `swap_enabled` and `engine_type`.
- Added `router/router_config.json.example` as a clean, hardware-agnostic template.
- Explicitly documented exclusion of vLLM, Aphrodite, ExLlamaV2/TabbyAPI, and hybrid offload engines (Strata) due to system RAM hogging conflicting with the Windows WDDM 50% pinned staging buffer.

---

## [2.2.0] - 2026-11-04

### Added
- **Strict VRAM Eviction Barrier**:
  - Integrated `enforce_vram_eviction()` in `router/vmm_router.py`.
  - Polling guard via `nvidia-smi` verifying physical VRAM drops below 3000 MB before any swap from RAM to VRAM can proceed.
  - Zero concurrent background CUDA initialization during active token decode, completely eliminating CUDA lock contention and preventing throughput degradation from 54 tok/s to <2 tok/s.
- **Closed-Loop 4-Stage Autonomous Pipeline**:
  - Stage 1 (Architect): `ATX-Swift-1.5 Q5_K_M` (27B Dense) synthesizes modular technical plan and contracts.
  - Stage 2 (Coder): `Gyro-Coder TQ2_0` (125B MoE / 3.8B Active) synthesizes production code in clean VRAM at >53.5 tok/s.
  - Stage 3 (Debugger & Security Auditor): `Cyber-ATX Q5_K_M` (27B Dense) audits code, analyzes runtime logs, and emits prioritized defect and vulnerability reports.
  - Stage 4 (Coder Fix Loop): Upon Debugger culmination, Debugger is immediately evicted (<50ms) and Coder (`Gyro-Coder TQ2_0`) is swapped into fresh VRAM (<2.3s) to correct and harden the code.
- **Windows WDDM High-Context Parameters (96K / 98,304 tokens)**:
  - Added `--load-mode none` to bypass Windows persistent `mmap` caching, preserving host RAM for `cudaHostAlloc` pinned mirror.
  - Fixed speculative MTP drafting with explicit `--spec-draft-vocab-map 65536` and `--spec-draft-vocab-hot 256`.
  - CPU thread distribution optimized to `-t 12 -tb 8` to eliminate thread oversubscription.
  - Context retention configured with `--repeat-last-n 256` and `--cache-ram 16384`.

### Changed
- `router/router_config.json`: Updated model roles, aliases, and launch parameters for strict 96k execution.
- `pipeline/autonomous_pipeline.py`: Upgraded to 4-stage sequential workflow with closed-loop Coder fix and code block extractors for Python and Kotlin.
- `router/vmm_router.py`: Upgraded version identifier to `2.2.0`. Automatic post-inference eviction for Debugger to prep clean VRAM for Coder.

---

## [2.1.0] - 2026-11-03

### Added
- **Concurrency & Lock Guard**:
  - Integrated `asyncio.Lock` inside `VMMStateManager` for thread-safe sequential swap transitions.
  - Model Context Protocol (MCP) server in `mcp/vmm_mcp_server.py` exposing `vmm_list_models`, `vmm_swap_model`, and `vmm_query_agent` for AI IDEs.
  - Alias resolution supporting common shorthand (`gyro`, `cyber`, `swift`, `coder`, `debugger`).
- **Benchmark Suite 2 (Full vs Incremental Debugger)**:
  - Documented 5.5x speedup (-81.8% latency, -91.9% prompt tokens) when using conditioned incremental debugging over full codebase re-emission.

### Changed
- Renamed project to `cuda-vmm-swap` to reflect engine-agnostic architecture.
- Enforced strict ASCII logging across all status messages and CLI output.

---

## [2.0.0] - 2026-11-02

### Added
- **Initial Release of cuda-vmm-swap**:
  - Native CUDA Virtual Memory Management (cuMem* Driver API) host-mirror swapping.
  - `<60 ms` physical VRAM release via `cuMemUnmap`.
  - `<2.3 s` PCIe 3.0 x16 DMA wake via `cuMemMap` + `cudaMemcpyAsync`.
  - Sliding Window Invariant documentation resolving Windows WDDM 24 GB pinned memory cap.
  - OpenAI-compatible FastAPI reverse proxy in `router/vmm_router.py`.
  - Autonomous 3-agent orchestration pipeline in `pipeline/autonomous_pipeline.py`.
