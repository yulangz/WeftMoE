CXX ?= c++
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD_DIR ?= build
CPPFLAGS := -Iinclude -Isrc -pthread
CUDA_CPPFLAGS := -Iinclude -Isrc -Xcompiler=-pthread
DEEPGEMM_INCLUDE_DIR ?= ../DeepGEMM/deep_gemm/include
# Prefer CUTLASS/CUTE bundled with an installed DeepGEMM package. A source-only
# build can override this with an initialized CUTLASS checkout.
CUTLASS_INCLUDE_DIR ?= $(shell python3 -c 'import importlib.util,os;s=importlib.util.find_spec("deep_gemm");print(os.path.join(os.path.dirname(s.origin),"include") if s else "../DeepGEMM/third-party/cutlass/include")')
CUDA_CPPFLAGS += -isystem $(DEEPGEMM_INCLUDE_DIR) -isystem $(CUTLASS_INCLUDE_DIR)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
WERROR ?= 0
MEGAMOE_DEBUG_VALIDATE ?= 0

CPPFLAGS += -DMEGAMOE_DEBUG_VALIDATE=$(MEGAMOE_DEBUG_VALIDATE)
CUDA_CPPFLAGS += -DMEGAMOE_DEBUG_VALIDATE=$(MEGAMOE_DEBUG_VALIDATE)

LIB_SOURCES := \
	src/cuda/control_protocol.cpp \
	src/metadata/notify_dispatch.cpp \
	src/metadata/serialize.cpp \
	src/transport/bootstrap_channel.cpp \
	src/transport/activation_protocol.cpp \
	src/transport/host_fake_count_exchange.cpp \
	src/scoreboard/workspace_plan.cpp
LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

CUDA_ARCH_FLAGS ?= -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a
CUDA_NVCCFLAGS ?= -O2 -g -std=c++17 --expt-relaxed-constexpr
ifeq ($(WERROR),1)
override CXXFLAGS += -Werror
override CUDA_NVCCFLAGS += -Xcompiler=-Werror
endif
ENABLE_CUDA ?= 0
NCCL_RUNTIME_LIB_DIR ?=
CUDA_RUN_ENV := $(if $(NCCL_RUNTIME_LIB_DIR),LD_LIBRARY_PATH=$(NCCL_RUNTIME_LIB_DIR):$${LD_LIBRARY_PATH},)
COMPUTE_SANITIZER ?= compute-sanitizer
CUDA_LIB_SOURCES := \
	src/cuda/device_notify.cu \
	src/cuda/activation_transport.cu \
	src/transport/nccl_count_exchange.cu
CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/%.o)

.PHONY: all test reference-test launcher-security-test cross-check header-test api-negative-test bench megatron-bench fair-integrated-bench cuda-lib cuda-test cuda-test-local cuda-protocol-test cuda-bench cuda-transport-test cuda-p2p-test cuda-p2p-bench cuda-integrated-bench cuda-backward-test cuda-backward-bench cuda-transport-bench cuda-sanitizer-smoke cuda-sm90-compile clean

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/metadata_dump $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -c $< -o $@

cuda-lib: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)

$(BUILD_DIR)/test_cuda_notify: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS) tests/test_cuda_notify.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(CUDA_NVCCFLAGS) $(CUDA_CPPFLAGS) $(LDFLAGS) $^ -lnccl -lcuda -ldl -o $@

cuda-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --local 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) python3 tests/run_control_protocol_regressions.py --binary $(BUILD_DIR)/test_cuda_notify

cuda-test-local: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --local 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --local 2
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --local 8

cuda-protocol-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_control_protocol_regressions.py --binary $(BUILD_DIR)/test_cuda_notify

cuda-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --bench 1

cuda-transport-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --transport-capability
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --device-smoke

cuda-p2p-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 2 --rounds 20
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 2 --rounds 20 --ring-slots 1
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 2 --rounds 20 --no-wrap
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 2 --cancel
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 8 --rounds 20

cuda-p2p-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 8 --benchmark --timeout 600

cuda-integrated-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(BUILD_DIR)/test_cuda_notify --world 8 --integrated-benchmark --timeout 900

cuda-backward-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(abspath $(BUILD_DIR)/test_cuda_notify) --world 1 --backward-smoke --timeout 300
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(abspath $(BUILD_DIR)/test_cuda_notify) --world 2 --backward-smoke --timeout 600
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(abspath $(BUILD_DIR)/test_cuda_notify) --world 8 --backward-smoke --timeout 900

cuda-backward-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py --binary $(abspath $(BUILD_DIR)/test_cuda_notify) --world 8 --backward-benchmark --backward-cache --rounds 110 --timeout 1800

megatron-bench:
	torchrun --standalone --nproc-per-node=8 benchmarks/megatron_dispatch_baseline.py

fair-integrated-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 benchmarks/run_integrated_comparison.py \
		--binary $(BUILD_DIR)/test_cuda_notify $(if $(NCCL_RUNTIME_LIB_DIR),--nccl-lib-dir $(NCCL_RUNTIME_LIB_DIR),)

cuda-transport-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --transport-bench

cuda-sanitizer-smoke: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool memcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool racecheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool initcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool memcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --forward-sanitizer
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool racecheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --forward-sanitizer
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool initcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --forward-sanitizer
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool memcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --backward-sanitizer
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool racecheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --backward-sanitizer
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool initcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --backward-sanitizer

cuda-sm90-compile:
	$(MAKE) BUILD_DIR=$(BUILD_DIR)-sm90 CUDA_ARCH_FLAGS='-gencode arch=compute_90a,code=sm_90a' cuda-lib

$(BUILD_DIR)/test_notify_dispatch: $(LIB_OBJECTS) tests/test_notify_dispatch.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/notify_dispatch_bench: $(LIB_OBJECTS) benchmarks/notify_dispatch_bench.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/metadata_dump: $(LIB_OBJECTS) tests/metadata_dump.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

reference-test:
	PYTHONPATH=. python3 -m unittest discover -v -s tests -p 'test_reference.py'

launcher-security-test:
	PYTHONPATH=. python3 tests/test_launcher_security.py

cross-check: $(BUILD_DIR)/metadata_dump
	$(BUILD_DIR)/metadata_dump > $(BUILD_DIR)/cpp_metadata.json
	PYTHONPATH=. python3 tests/compare_reference.py $(BUILD_DIR)/cpp_metadata.json

header-test:
	CXX='$(CXX)' python3 tests/check_headers.py

api-negative-test:
	CXX='$(CXX)' python3 tests/check_cuda_public_api.py

test: $(BUILD_DIR)/test_notify_dispatch reference-test launcher-security-test cross-check header-test api-negative-test
	$(BUILD_DIR)/test_notify_dispatch

ifeq ($(ENABLE_CUDA),1)
test: cuda-test
endif

bench: $(BUILD_DIR)/notify_dispatch_bench
	$(BUILD_DIR)/notify_dispatch_bench

clean:
	rm -rf -- $(BUILD_DIR)
