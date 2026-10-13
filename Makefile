# Threads per value: uint32 atomicAdd vs. spinlock, shared vs. global memory.
# One source (threads_per_value.cu) for NVIDIA (nvcc) and AMD (hipcc) GPUs, and a
# port for Apple GPUs (threads_per_value.metal + threads_per_value_metal.cpp, metal-cpp).
#
#   NVIDIA (CUDA)     AMD (HIP / ROCm)    Apple (Metal)
#   make              make hip            make metal          build for the GPU in this machine
#   make test         make test-hip       make test-metal     exact correctness check
#   make run          make run-hip        make run-metal      benchmark + plot -> results/, results_hip/, results_metal/
#
#   make plot     re-plot results/      (make plot RESULTS=results_hip for the AMD results)
#   make venv     create .venv with matplotlib for the plot
#   make clean    remove the binaries
#
# Override the target GPU with ARCH (NVIDIA, e.g. ARCH=sm_86) or HIP_ARCH (AMD, e.g. HIP_ARCH=gfx90a).
# The Metal build needs the metal-cpp headers: METAL_CPP=/path/to/metal-cpp.
NVCC        ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
HIPCC       ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ARCH        ?= native
HIP_ARCH    ?= native
FLAGS       ?= -O3 -std=c++17
METAL_CPP   ?= $(HOME)/lingodb-gpu/metal-cpp
RESULTS     ?= results
RESULTS_HIP ?= results_hip
RESULTS_METAL ?= results_metal
PY          := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

threads_per_value: threads_per_value.cu
	$(NVCC) $(FLAGS) -lineinfo -arch=$(ARCH) -o $@ $<

threads_per_value_hip: threads_per_value.cu
	$(HIPCC) $(FLAGS) -x hip --offload-arch=$(HIP_ARCH) -o $@ $<

hip: threads_per_value_hip

# MSL 4.1: scoped atomics (atomic<uint, thread_scope_threadgroup>)
threads_per_value.metallib: threads_per_value.metal
	xcrun -sdk macosx metal -std=metal4.1 -O3 -o $@ $<

threads_per_value_metal: threads_per_value_metal.cpp threads_per_value.metallib
	clang++ $(FLAGS) -I$(METAL_CPP) -o $@ $< -framework Metal -framework Foundation -framework IOKit -framework CoreFoundation

metal: threads_per_value_metal

test: threads_per_value
	./threads_per_value --test

test-hip: threads_per_value_hip
	./threads_per_value_hip --test

test-metal: threads_per_value_metal
	./threads_per_value_metal --test

run: threads_per_value
	@mkdir -p $(RESULTS)
	./threads_per_value --out $(RESULTS) | tee $(RESULTS)/run.log
	$(PY) plot.py $(RESULTS)

run-hip: threads_per_value_hip
	@mkdir -p $(RESULTS_HIP)
	./threads_per_value_hip --out $(RESULTS_HIP) | tee $(RESULTS_HIP)/run.log
	$(PY) plot.py $(RESULTS_HIP)

run-metal: threads_per_value_metal
	@mkdir -p $(RESULTS_METAL)
	./threads_per_value_metal --out $(RESULTS_METAL) | tee $(RESULTS_METAL)/run.log
	$(PY) plot.py $(RESULTS_METAL)

plot:
	$(PY) plot.py $(RESULTS)

venv:
	python3 -m venv .venv && .venv/bin/pip install matplotlib

clean:
	rm -f threads_per_value threads_per_value_hip threads_per_value_metal threads_per_value.metallib

.PHONY: hip metal test test-hip test-metal run run-hip run-metal plot venv clean
