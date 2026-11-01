@echo off
REM Build and run all llamampere-vmm-swap benchmarks
REM Requires: Visual Studio Build Tools 2022 + CUDA Toolkit (nvcc)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo === Building test_vmm_15gb_driver (Full VMM benchmark, native Driver API) ===
nvcc -O3 -lcuda benchmarks\test_vmm_15gb_driver.cu -o benchmarks\test_vmm_15gb_driver.exe
if %ERRORLEVEL% EQU 0 (
    echo Running...
    benchmarks\test_vmm_15gb_driver.exe
) else (
    echo BUILD FAILED: test_vmm_15gb_driver
)

echo.
echo === Building test_vmm_swap (VMM + VirtualAlloc) ===
nvcc -O3 -lcuda benchmarks\test_vmm_swap.cu -o benchmarks\test_vmm_swap.exe
if %ERRORLEVEL% EQU 0 (
    echo Running...
    benchmarks\test_vmm_swap.exe
) else (
    echo BUILD FAILED: test_vmm_swap
)

echo.
echo === Building test_pipeline (Double-buffered pipelined transfer) ===
nvcc -O3 -lcuda benchmarks\test_pipeline.cu -o benchmarks\test_pipeline.exe
if %ERRORLEVEL% EQU 0 (
    echo Running...
    benchmarks\test_pipeline.exe
) else (
    echo BUILD FAILED: test_pipeline
)

echo.
echo === Building test_chunked_pinned (Chunked cudaHostAlloc) ===
nvcc -O3 benchmarks\test_chunked_pinned.cpp -o benchmarks\test_chunked_pinned.exe
if %ERRORLEVEL% EQU 0 (
    echo Running...
    benchmarks\test_chunked_pinned.exe
) else (
    echo BUILD FAILED: test_chunked_pinned
)

echo.
echo === Building test_host_mirror (VirtualAlloc + cudaHostRegister) ===
nvcc -O3 benchmarks\test_host_mirror.cpp -o benchmarks\test_host_mirror.exe
if %ERRORLEVEL% EQU 0 (
    echo Running...
    benchmarks\test_host_mirror.exe
) else (
    echo BUILD FAILED: test_host_mirror
)

echo.
echo === All benchmarks done ===
pause
