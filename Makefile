CXX ?= c++
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD_DIR ?= build
CPPFLAGS := -Iinclude -Isrc -Itests -pthread
CUDA_CPPFLAGS := -Iinclude -Isrc -Itests -Xcompiler=-pthread
DEEPGEMM_INCLUDE_DIR ?= ../DeepGEMM/deep_gemm/include
# Prefer CUTLASS/CUTE bundled with an installed DeepGEMM package. A source-only
# build can override this with an initialized CUTLASS checkout.
CUTLASS_INCLUDE_DIR ?= $(shell python3 -c 'import importlib.util,os;s=importlib.util.find_spec("deep_gemm");print(os.path.join(os.path.dirname(s.origin),"include") if s else "../DeepGEMM/third-party/cutlass/include")')
CUDA_CPPFLAGS += -isystem $(DEEPGEMM_INCLUDE_DIR) -isystem $(CUTLASS_INCLUDE_DIR)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
override CXXFLAGS += -fPIC
WERROR ?= 0
MEGAMOE_DEBUG_VALIDATE ?= 0
PREFIX ?= /usr/local
DESTDIR ?=
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
DATADIR ?= $(PREFIX)/share/megamoe
LIBMEGAMOE_VERSION := 2.0.0
LIBMEGAMOE_SONAME := libmegamoe.so.2
LIBMEGAMOE_REALNAME := libmegamoe.so.$(LIBMEGAMOE_VERSION)

PUBLIC_HEADERS := \
	include/megamoe/version.hpp \
	include/megamoe/api/errors.hpp \
	include/megamoe/api/notify_dispatch.hpp \
	include/megamoe/config/runtime_config.hpp \
	include/megamoe/cuda/activation_transport.hpp \
	include/megamoe/cuda/control_protocol.hpp \
	include/megamoe/cuda/device_notify.hpp \
	include/megamoe/cuda/nccl_count_exchange.hpp \
	include/megamoe/metadata/dispatch_handle.hpp \
	include/megamoe/scoreboard/workspace_plan.hpp \
	include/megamoe/transport/activation_protocol.hpp \
	include/megamoe/transport/count_exchange.hpp

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
CUDA_NVCCFLAGS ?= -O2 -g -std=c++17 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
override CUDA_NVCCFLAGS += -Xcompiler=-fPIC
ifeq ($(WERROR),1)
override CXXFLAGS += -Werror
override CUDA_NVCCFLAGS += -Xcompiler=-Werror
endif
ENABLE_CUDA ?= 0
NCCL_RUNTIME_LIB_DIR ?=
CUDA_RUN_ENV := $(if $(NCCL_RUNTIME_LIB_DIR),LD_LIBRARY_PATH=$(NCCL_RUNTIME_LIB_DIR):$${LD_LIBRARY_PATH},)
COMPUTE_SANITIZER ?= compute-sanitizer
SANITIZER_PRELOAD ?=
CROSS_CHECK_ENV := $(if $(SANITIZER_PRELOAD),LD_PRELOAD=$(SANITIZER_PRELOAD),)
CUDA_LIB_SOURCES := \
	src/cuda/device_notify.cu \
	src/cuda/activation_transport.cu \
	src/transport/nccl_count_exchange.cu
CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/%.o)

.PHONY: all test reference-test launcher-security-test cross-check header-test api-negative-test shared-lib install uninstall-manifest install-test kernel-artifact-check kernel-artifact-update release-test bench megatron-bench fair-integrated-bench fair-backward-bench dw1-roofline-bench cuda-lib cuda-test cuda-runtime-shape-test cuda-test-local cuda-protocol-test cuda-bench cuda-transport-test cuda-p2p-test cuda-p2p-bench cuda-integrated-bench cuda-backward-test cuda-backward-bench cuda-transport-bench cuda-sanitizer-smoke cuda-sm90-compile clean

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/metadata_dump $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -c $< -o $@

$(BUILD_DIR)/src/cuda/activation_transport.o: \
	src/cuda/forward/kernel_contract.cuh \
	src/cuda/forward/runtime_shape_fallback.cuh \
	src/cuda/backward/kernel_contract.cuh

cuda-lib: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)

$(BUILD_DIR)/$(LIBMEGAMOE_REALNAME): $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) -Wno-deprecated-gpu-targets -shared \
		-Xlinker=-soname -Xlinker=$(LIBMEGAMOE_SONAME) \
		$(LDFLAGS) $^ -lnccl -lcudart -lcuda -ldl -o $@

shared-lib: $(BUILD_DIR)/$(LIBMEGAMOE_REALNAME)

kernel-artifact-check: $(BUILD_DIR)/src/cuda/activation_transport.o
	python3 tools/kernel_artifact_manifest.py \
		--binary $< --manifest tools/kernel_artifact_manifest.json

kernel-artifact-update: $(BUILD_DIR)/src/cuda/activation_transport.o
	python3 tools/kernel_artifact_manifest.py \
		--binary $< --manifest tools/kernel_artifact_manifest.json --update

release-test: shared-lib kernel-artifact-check install-test

CUDA_TEST_SOURCES := \
	tests/test_cuda_notify.cpp \
	tests/cuda/runtime_shape_tests.cpp

$(BUILD_DIR)/test_cuda_notify: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS) $(CUDA_TEST_SOURCES)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(CUDA_NVCCFLAGS) $(CUDA_CPPFLAGS) $(LDFLAGS) $^ -lnccl -lcuda -ldl -o $@

cuda-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --local 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(MAKE) BUILD_DIR=$(BUILD_DIR) cuda-runtime-shape-test
	$(CUDA_RUN_ENV) python3 tests/run_control_protocol_regressions.py --binary $(BUILD_DIR)/test_cuda_notify

cuda-runtime-shape-test: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --runtime-shape 64 32 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --runtime-shape 96 48 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --runtime-shape 127 65 1
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --runtime-shape 256 192 1
	$(CUDA_RUN_ENV) python3 tests/run_local_activation_p2p.py \
		--binary $(abspath $(BUILD_DIR)/test_cuda_notify) --world 2 \
		--backward-smoke --runtime-hidden 127 --runtime-intermediate 65 \
		--backward-smoke-repeats 1 --timeout 180

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

# Usage: make fair-backward-bench MEGAMOE_RUNS='run1.log run2.log run3.log' \
#   MEGATRON_RUN=megatron.log
fair-backward-bench:
	python3 benchmarks/compare_backward_training.py \
		--megamoe $(MEGAMOE_RUNS) --megatron $(MEGATRON_RUN)

dw1-roofline-bench:
	python3 benchmarks/dw1_roofline_baseline.py

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
	$(CROSS_CHECK_ENV) $(BUILD_DIR)/metadata_dump > $(BUILD_DIR)/cpp_metadata.json
	PYTHONPATH=. python3 tests/compare_reference.py $(BUILD_DIR)/cpp_metadata.json

header-test:
	CXX='$(CXX)' python3 tests/check_headers.py

api-negative-test:
	CXX='$(CXX)' python3 tests/check_cuda_public_api.py

install: shared-lib
	@for header in $(PUBLIC_HEADERS); do \
		destination='$(DESTDIR)$(INCLUDEDIR)/'$${header#include/}; \
		install -D -m 0644 $$header $$destination; \
	done
	install -D -m 0755 $(BUILD_DIR)/$(LIBMEGAMOE_REALNAME) \
		$(DESTDIR)$(LIBDIR)/$(LIBMEGAMOE_REALNAME)
	ln -sfn $(LIBMEGAMOE_REALNAME) $(DESTDIR)$(LIBDIR)/$(LIBMEGAMOE_SONAME)
	ln -sfn $(LIBMEGAMOE_SONAME) $(DESTDIR)$(LIBDIR)/libmegamoe.so
	@mkdir -p $(DESTDIR)$(PKGCONFIGDIR)
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(LIBMEGAMOE_VERSION)|g' \
		packaging/megamoe.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/megamoe.pc
	install -D -m 0644 examples/minimal_training.cpp \
		$(DESTDIR)$(DATADIR)/examples/minimal_training.cpp
	install -D -m 0644 LICENSE $(DESTDIR)$(DATADIR)/LICENSE
	install -D -m 0644 NOTICE $(DESTDIR)$(DATADIR)/NOTICE
	install -D -m 0644 THIRD_PARTY_NOTICES.md \
		$(DESTDIR)$(DATADIR)/THIRD_PARTY_NOTICES.md
	@{ \
		for header in $(PUBLIC_HEADERS); do echo '$(INCLUDEDIR)/'$${header#include/}; done; \
		echo '$(LIBDIR)/$(LIBMEGAMOE_REALNAME)'; \
		echo '$(LIBDIR)/$(LIBMEGAMOE_SONAME)'; \
		echo '$(LIBDIR)/libmegamoe.so'; \
		echo '$(PKGCONFIGDIR)/megamoe.pc'; \
		echo '$(DATADIR)/examples/minimal_training.cpp'; \
		echo '$(DATADIR)/LICENSE'; \
		echo '$(DATADIR)/NOTICE'; \
		echo '$(DATADIR)/THIRD_PARTY_NOTICES.md'; \
		echo '$(DATADIR)/install_manifest.txt'; \
	} > $(DESTDIR)$(DATADIR)/install_manifest.txt

uninstall-manifest:
	@printf '%s\n' 'Installed files are listed in $(DATADIR)/install_manifest.txt.' \
		'This target is intentionally non-destructive; remove exactly that manifest.'

install-test:
	CXX='$(CXX)' python3 tests/check_install.py

test: $(BUILD_DIR)/test_notify_dispatch reference-test launcher-security-test cross-check header-test api-negative-test
	$(BUILD_DIR)/test_notify_dispatch

ifeq ($(ENABLE_CUDA),1)
test: cuda-test
endif

bench: $(BUILD_DIR)/notify_dispatch_bench
	$(BUILD_DIR)/notify_dispatch_bench

clean:
	rm -rf -- $(BUILD_DIR)
