CXX ?= c++
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD_DIR ?= build
CPPFLAGS := -Iinclude -Isrc -Itests -pthread
CUDA_CPPFLAGS := -Iinclude -Isrc -Itests -Xcompiler=-pthread
# The exact MIT-licensed DeepGEMM headers used by the frozen SM90 kernels are
# kept in-tree so a git archive is a complete source release. CUTLASS/CUTE
# remains an external build dependency and is discovered from the installed
# DeepGEMM package unless explicitly overridden.
DEEPGEMM_INCLUDE_DIR ?= src/cuda/vendor
CUTLASS_INCLUDE_DIR ?= $(shell python3 -c 'import importlib.util,os;s=importlib.util.find_spec("deep_gemm");print(os.path.join(os.path.dirname(s.origin),"include") if s else "../DeepGEMM/third-party/cutlass/include")')
CUDA_CPPFLAGS += -isystem $(DEEPGEMM_INCLUDE_DIR) -isystem $(CUTLASS_INCLUDE_DIR)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
override CXXFLAGS += -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections
WERROR ?= 0
MEGAMOE_DEBUG_VALIDATE ?= 0
MEGAMOE_DEVICE_TIMELINE ?= 0
MEGAMOE_FORWARD_LIGHT_PROGRESS ?= 0
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
	include/megamoe/cuda/backward.hpp \
	include/megamoe/cuda/control_protocol.hpp \
	include/megamoe/cuda/device_notify.hpp \
	include/megamoe/cuda/nccl_count_exchange.hpp \
	include/megamoe/metadata/dispatch_handle.hpp \
	include/megamoe/scoreboard/workspace_plan.hpp \
	include/megamoe/transport/activation_protocol.hpp \
	include/megamoe/transport/count_exchange.hpp

CPPFLAGS += -DMEGAMOE_DEBUG_VALIDATE=$(MEGAMOE_DEBUG_VALIDATE) -DMEGAMOE_BUILD_SHARED=1
CUDA_CPPFLAGS += -DMEGAMOE_DEBUG_VALIDATE=$(MEGAMOE_DEBUG_VALIDATE) -DMEGAMOE_BUILD_SHARED=1
CPPFLAGS += -DMEGAMOE_DEVICE_TIMELINE=$(MEGAMOE_DEVICE_TIMELINE)
CUDA_CPPFLAGS += -DMEGAMOE_DEVICE_TIMELINE=$(MEGAMOE_DEVICE_TIMELINE)
CPPFLAGS += -DMEGAMOE_FORWARD_LIGHT_PROGRESS=$(MEGAMOE_FORWARD_LIGHT_PROGRESS)
CUDA_CPPFLAGS += -DMEGAMOE_FORWARD_LIGHT_PROGRESS=$(MEGAMOE_FORWARD_LIGHT_PROGRESS)
LIB_SOURCES := \
	src/cuda/backward_stub.cpp \
	src/cuda/control_protocol.cpp \
	src/metadata/notify_dispatch.cpp \
	src/metadata/serialize.cpp \
	src/transport/bootstrap_channel.cpp \
	src/transport/activation_protocol.cpp \
	src/transport/host_fake_count_exchange.cpp \
	src/scoreboard/workspace_plan.cpp
LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
PROD_LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/production/%.o)

CUDA_ARCH_FLAGS ?= -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a
CUDA_NVCCFLAGS ?= -O2 -g -std=c++17 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
override CUDA_NVCCFLAGS += -Xcompiler=-fPIC,-fvisibility=hidden,-ffunction-sections,-fdata-sections
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
	src/cuda/persistent_v3/kernel_entry.cu \
	src/transport/nccl_count_exchange.cu
CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/%.o) \
	$(BUILD_DIR)/src/cuda/forward_transport.o
PROD_CUDA_LIB_OBJECTS := \
	$(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/production/%.o) \
	$(BUILD_DIR)/production/src/cuda/forward_transport.o

.PHONY: all test reference-test launcher-security-test forward-autotune-test forward-autotune cross-check header-test api-negative-test shared-lib testing-lib install uninstall-manifest install-test installed-header-scan release-text-check public-symbol-check retired-symbol-link-negative kernel-artifact-check kernel-artifact-update license-check release-test release-check release-check-inner bench megatron-bench megatron-backward-bench fair-integrated-bench cuda-lib cuda-test cuda-test-local cuda-protocol-test cuda-bench cuda-transport-test cuda-p2p-test cuda-p2p-bench cuda-integrated-bench cuda-transport-bench cuda-sanitizer-smoke cuda-sm90-compile clean

forward-autotune-test:
	python3 tests/test_forward_autotune.py

forward-autotune:
	python3 tools/autotune_forward.py --config configs/forward_h200_autotune.json

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/metadata_dump $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -DMEGAMOE_BUILD_TESTING=1 $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DMEGAMOE_BUILD_TESTING=1 $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -c $< -o $@

$(BUILD_DIR)/production/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -DMEGAMOE_BUILD_TESTING=0 $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/production/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DMEGAMOE_BUILD_TESTING=0 $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -c $< -o $@

$(BUILD_DIR)/src/cuda/activation_transport.o \
$(BUILD_DIR)/production/src/cuda/activation_transport.o: \
	src/cuda/forward/kernel_contract.cuh \
	src/cuda/forward/kernel_support.cuh \
	src/cuda/forward/device_kernels.cuh \
	src/cuda/backward/kernel_contract.cuh \
	src/cuda/backward/device_kernels.cuh \
	src/cuda/backward/tuning_config.hpp \
	src/cuda/persistent_v3/backward/compute_kernels.cuh \
	src/cuda/backward/host_launch.inc \
	src/cuda/forward/runtime_state.inc \
	src/cuda/forward/host_launch.inc \
	src/cuda/testing/transport_probes.inc \
	src/cuda/forward/tuning_config.hpp \
	src/cuda/forward/transport_launch.hpp \
	src/cuda/persistent_v3/contracts.hpp \
	src/cuda/persistent_v3/queues.cuh \
	src/cuda/persistent_v3/roles/common.cuh \
	src/cuda/persistent_v3/forward/pipeline_state.cuh \
	src/cuda/persistent_v3/forward/dispatch_pipeline.cuh \
	src/cuda/persistent_v3/forward/return_pipeline.cuh \
	src/cuda/persistent_v3/forward/finalize_pipeline.cuh \
	src/cuda/persistent_v3/forward/compute_kernel.cuh \
	src/cuda/persistent_v3/forward/reduce_pipeline.cuh \
	src/cuda/profiling/device_timeline.cuh \
	src/cuda/profiling/device_timeline_types.hpp

$(BUILD_DIR)/src/cuda/forward_transport.o \
$(BUILD_DIR)/production/src/cuda/forward_transport.o: \
	src/cuda/activation_transport.cu \
	src/cuda/activation_transport_internal.hpp \
	src/cuda/cuda_internal.hpp \
	src/cuda/forward/kernel_contract.cuh \
	src/cuda/forward/kernel_support.cuh \
	src/cuda/forward/device_kernels.cuh \
	src/cuda/forward/runtime_state.inc \
	src/cuda/forward/host_launch.inc \
	src/cuda/testing/transport_probes.inc \
	src/cuda/forward/tuning_config.hpp \
	src/cuda/forward/transport_launch.hpp \
	src/cuda/persistent_v3/contracts.hpp \
	src/cuda/persistent_v3/queues.cuh \
	src/cuda/persistent_v3/roles/common.cuh \
	src/cuda/persistent_v3/forward/pipeline_state.cuh \
	src/cuda/persistent_v3/forward/reduce_queue.cuh \
	src/cuda/persistent_v3/forward/dispatch_pipeline.cuh \
	src/cuda/persistent_v3/forward/return_pipeline.cuh \
	src/cuda/persistent_v3/forward/finalize_pipeline.cuh \
	src/cuda/persistent_v3/forward/compute_kernel.cuh \
	src/cuda/persistent_v3/forward/reduce_pipeline.cuh \
	src/cuda/profiling/device_timeline.cuh \
	src/cuda/profiling/device_timeline_types.hpp \
	src/cuda/transport_generation_completion.cuh

$(BUILD_DIR)/src/cuda/forward_transport.o:
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DMEGAMOE_BUILD_TESTING=1 \
		-DMEGAMOE_FORWARD_TRANSPORT_TU=1 -diag-suppress=177 $(CUDA_NVCCFLAGS) \
		$(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror \
		-c src/cuda/activation_transport.cu -o $@

$(BUILD_DIR)/production/src/cuda/forward_transport.o:
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DMEGAMOE_BUILD_TESTING=0 \
		-DMEGAMOE_FORWARD_TRANSPORT_TU=1 -diag-suppress=177 $(CUDA_NVCCFLAGS) \
		$(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror \
		-c src/cuda/activation_transport.cu -o $@

$(BUILD_DIR)/src/cuda/persistent_v3/kernel_entry.o \
$(BUILD_DIR)/production/src/cuda/persistent_v3/kernel_entry.o: \
	src/cuda/persistent_v3/contracts.hpp \
	src/cuda/persistent_v3/epoch_state.cuh \
	src/cuda/persistent_v3/queues.cuh \
	src/cuda/persistent_v3/shared_storage.cuh \
	src/cuda/persistent_v3/pipelines/task_scoreboard.cuh \
	src/cuda/persistent_v3/roles/common.cuh \
	src/cuda/persistent_v3/roles/compute_role.cuh \
	src/cuda/backward/tuning_config.hpp \
	src/cuda/persistent_v3/backward/scheduler.cuh \
	src/cuda/persistent_v3/backward/scheduler_probe.cuh \
	src/cuda/persistent_v3/testing.hpp

cuda-lib: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)

$(BUILD_DIR)/$(LIBMEGAMOE_REALNAME): $(PROD_LIB_OBJECTS) $(PROD_CUDA_LIB_OBJECTS) abi/libmegamoe.map
	@mkdir -p $(@D)
	$(NVCC) -Wno-deprecated-gpu-targets -shared \
		-Xlinker=-soname -Xlinker=$(LIBMEGAMOE_SONAME) \
		-Xlinker=--version-script=$(CURDIR)/abi/libmegamoe.map \
		-Xlinker=--gc-sections \
		$(LDFLAGS) $(filter %.o,$^) -lnccl -lcublas -lcudart -lcuda -ldl -o $@

shared-lib: $(BUILD_DIR)/$(LIBMEGAMOE_REALNAME)

$(BUILD_DIR)/libmegamoe_testing.a: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)
	@mkdir -p $(@D)
	ar rcs $@ $^

# Repository tests may link this non-installed archive. Production packages
# only build the MEGAMOE_BUILD_TESTING=0 object graph above.
testing-lib: $(BUILD_DIR)/libmegamoe_testing.a

kernel-artifact-check: $(BUILD_DIR)/production/src/cuda/activation_transport.o
	python3 tools/kernel_artifact_manifest.py \
		--binary $< --manifest tools/kernel_artifact_manifest.json

kernel-artifact-update: $(BUILD_DIR)/production/src/cuda/activation_transport.o
	python3 tools/kernel_artifact_manifest.py \
		--binary $< --manifest tools/kernel_artifact_manifest.json --update

installed-header-scan:
	python3 tools/check_release_hygiene.py --root . --self-test

release-text-check:
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "release text check: mode=git"; \
		git diff --check; \
	else \
		echo "release text check: mode=archive"; \
		python3 tools/check_release_hygiene.py --root . --self-test; \
	fi

public-symbol-check: shared-lib
	python3 tools/check_public_symbols.py --binary $(BUILD_DIR)/$(LIBMEGAMOE_REALNAME) \
		--whitelist abi/public_symbols_v2.txt

retired-symbol-link-negative: shared-lib
	ln -sfn $(LIBMEGAMOE_REALNAME) $(BUILD_DIR)/libmegamoe.so
	python3 tools/check_retired_symbol_link.py --library-dir $(abspath $(BUILD_DIR))

license-check:
	test -s LICENSE
	test -s NOTICE
	test -s THIRD_PARTY_NOTICES.md

release-test: shared-lib kernel-artifact-check install-test

release-check:
	$(MAKE) BUILD_DIR=$(BUILD_DIR)-release clean
	$(MAKE) BUILD_DIR=$(BUILD_DIR)-release ENABLE_CUDA=1 \
		NCCL_RUNTIME_LIB_DIR='$(NCCL_RUNTIME_LIB_DIR)' release-check-inner

release-check-inner:
	+$(MAKE) BUILD_DIR=$(BUILD_DIR) ENABLE_CUDA=1 shared-lib
	+$(MAKE) BUILD_DIR=$(BUILD_DIR) installed-header-scan public-symbol-check \
		retired-symbol-link-negative kernel-artifact-check header-test \
		api-negative-test license-check release-text-check
	+$(MAKE) BUILD_DIR=$(BUILD_DIR) ENABLE_CUDA=1 \
		NCCL_RUNTIME_LIB_DIR='$(NCCL_RUNTIME_LIB_DIR)' install-test
	+$(MAKE) BUILD_DIR=$(BUILD_DIR)-werror ENABLE_CUDA=1 WERROR=1 shared-lib

CUDA_TEST_SOURCES := \
	tests/test_cuda_notify.cpp
CUDA_TEST_DEPS := \
	tests/cuda/backward_runner.inc \
	tests/cuda/distributed_runner.inc \
	tests/cuda/transport_benchmarks.inc

$(BUILD_DIR)/test_cuda_notify: $(BUILD_DIR)/libmegamoe_testing.a $(CUDA_TEST_SOURCES) $(CUDA_TEST_DEPS)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(CUDA_NVCCFLAGS) $(CUDA_CPPFLAGS) $(LDFLAGS) \
		$(CUDA_TEST_SOURCES) -Xlinker=--whole-archive \
		$(BUILD_DIR)/libmegamoe_testing.a -Xlinker=--no-whole-archive \
		-lnccl -lcublas -lcuda -ldl -o $@

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

megatron-bench:
	torchrun --standalone --nproc-per-node=8 benchmarks/megatron_dispatch_baseline.py

megatron-backward-bench:
	torchrun --standalone --nproc-per-node=8 \
		benchmarks/megatron_backward_baseline.py

fair-integrated-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) python3 benchmarks/run_integrated_comparison.py \
		--binary $(BUILD_DIR)/test_cuda_notify $(if $(NCCL_RUNTIME_LIB_DIR),--nccl-lib-dir $(NCCL_RUNTIME_LIB_DIR),)

cuda-transport-bench: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(BUILD_DIR)/test_cuda_notify --transport-bench

cuda-sanitizer-smoke: $(BUILD_DIR)/test_cuda_notify
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool memcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool racecheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke
	$(CUDA_RUN_ENV) $(COMPUTE_SANITIZER) --tool initcheck --error-exitcode=1 $(BUILD_DIR)/test_cuda_notify --device-smoke

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
	install -D -m 0644 examples/minimal_forward.cpp \
		$(DESTDIR)$(DATADIR)/examples/minimal_forward.cpp
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
		echo '$(DATADIR)/examples/minimal_forward.cpp'; \
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
