---
name: vmm-swap
description: Controls the local CUDA Virtual Memory Management (VMM) fast hot-swap engine (cuda-vmm-swap) across multi-agent roles (architect, coder, debugger) in sub-2.3 seconds with strict Windows WDDM memory limits.
---

# CUDA VMM Hot-Swap Skill

Use this skill when running local multi-agent software engineering workflows requiring specialized models on a single GPU.

## Supported Roles & Endpoints

The VMM Router exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1` with three primary roles:
* **`architect`**: High-context systemic software design, interface specifications, and unit test contracts.
* **`coder`**: High-throughput code synthesis, algorithm implementation, and unit test suites.
* **`debugger`**: Forensic defect isolation, compiler traceback diagnosis, and minimal surgical patch generation.

## How to Trigger Hot-Swapping

### 1. Transparent Model Switching (Recommended)
Simply specify the model role in your completion requests:
```python
import urllib.request, json

payload = {
    "model": "coder",  # Automatically triggers eviction of previous model and <2.3s DMA wake
    "messages": [{"role": "user", "content": "Implement the calculation engine"}],
    "temperature": 0.6
}
```

### 2. Manual VMM Hot-Swap via HTTP
```bash
curl -X POST http://127.0.0.1:8000/v1/vmm/swap -H "Content-Type: application/json" -d '{"model": "coder"}'
```

### 3. Querying Hardware & Memory Budgets
```bash
curl http://127.0.0.1:8000/status
```

## Hardware Invariants
* **Strict Eviction Rule**: The previous model is 100% unmapped (`cuMemUnmap` in <60ms) and verified below 3000 MB physical VRAM before the target model is mapped.
* **Windows WDDM Pinned Memory Cap**: Host RAM staging (`cudaHostAlloc`) is strictly limited to 50% of physical system RAM. Never attempt to stage multiple 18+ GB models simultaneously in RAM.
