# Threads per value: uint32 atomicAdd vs. spinlock, shared vs. global memory.
# One source (threads_per_value.cu) for NVIDIA (nvcc) and AMD (hipcc) GPUs.
#
#   NVIDIA (CUDA)                 AMD (HIP / ROCm)
#   make                          make hip              build for the GPU in this machine
#   make test                     make test-hip         exact correctness check (~25 s)
#   make run                      make run-hip          benchmark + plot -> results/ or results_hip/
#
#   make plot     re-plot results/      (make plot RESULTS=results_hip for the AMD results)
#   make venv     create .venv with matplotlib for the plot
#   make clean    remove the binaries
#
# Override the target GPU with ARCH (NVIDIA, e.g. ARCH=sm_86) or HIP_ARCH (AMD, e.g. HIP_ARCH=gfx90a).
NVCC        ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
HIPCC       ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ARCH        ?= native
HIP_ARCH    ?= native
FLAGS       ?= -O3 -std=c++17
RESULTS     ?= results
RESULTS_HIP ?= results_hip
PY          := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

threads_per_value: threads_per_value.cu
	$(NVCC) $(FLAGS) -lineinfo -arch=$(ARCH) -o $@ $<

threads_per_value_hip: threads_per_value.cu
	$(HIPCC) $(FLAGS) -x hip --offload-arch=$(HIP_ARCH) -o $@ $<

hip: threads_per_value_hip

test: threads_per_value
	./threads_per_value --test

test-hip: threads_per_value_hip
	./threads_per_value_hip --test

run: threads_per_value
	@mkdir -p $(RESULTS)
	./threads_per_value --out $(RESULTS) | tee $(RESULTS)/run.log
	$(PY) plot.py $(RESULTS)

run-hip: threads_per_value_hip
	@mkdir -p $(RESULTS_HIP)
	./threads_per_value_hip --out $(RESULTS_HIP) | tee $(RESULTS_HIP)/run.log
	$(PY) plot.py $(RESULTS_HIP)

plot:
	$(PY) plot.py $(RESULTS)

venv:
	python3 -m venv .venv && .venv/bin/pip install matplotlib

clean:
	rm -f threads_per_value threads_per_value_hip

.PHONY: hip test test-hip run run-hip plot venv clean
