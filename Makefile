CXX ?= c++
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD_DIR ?= build
CPPFLAGS := -Iinclude -Isrc -Itests -pthread
CUDA_CPPFLAGS := -Iinclude -Isrc -Itests -Xcompiler=-pthread
# Use a standalone CUTLASS checkout/install; no DeepGEMM Python package is needed.
CUTLASS_INCLUDE_DIR ?= /opt/cutlass/include
CUDA_CPPFLAGS += -isystem $(CUTLASS_INCLUDE_DIR)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
override CXXFLAGS += -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections
WERROR ?= 0
WEFTMOE_DEBUG_VALIDATE ?= 0
WEFTMOE_DEVICE_TIMELINE ?= 0
WEFTMOE_FORWARD_LIGHT_PROGRESS ?= 0
PREFIX ?= /usr/local
DESTDIR ?=
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
DATADIR ?= $(PREFIX)/share/weftmoe
LIBWEFTMOE_VERSION := 2.0.0
LIBWEFTMOE_SONAME := libweftmoe.so.2
LIBWEFTMOE_REALNAME := libweftmoe.so.$(LIBWEFTMOE_VERSION)

PUBLIC_HEADERS := \
	include/weftmoe/version.hpp \
	include/weftmoe/api/errors.hpp \
	include/weftmoe/api/notify_dispatch.hpp \
	include/weftmoe/config/runtime_config.hpp \
	include/weftmoe/cuda/activation_transport.hpp \
	include/weftmoe/cuda/backward.hpp \
	include/weftmoe/cuda/control_protocol.hpp \
	include/weftmoe/cuda/device_notify.hpp \
	include/weftmoe/cuda/nccl_count_exchange.hpp \
	include/weftmoe/metadata/dispatch_handle.hpp \
	include/weftmoe/scoreboard/workspace_plan.hpp \
	include/weftmoe/transport/activation_protocol.hpp \
	include/weftmoe/transport/count_exchange.hpp

CPPFLAGS += -DWEFTMOE_DEBUG_VALIDATE=$(WEFTMOE_DEBUG_VALIDATE) -DWEFTMOE_BUILD_SHARED=1
CUDA_CPPFLAGS += -DWEFTMOE_DEBUG_VALIDATE=$(WEFTMOE_DEBUG_VALIDATE) -DWEFTMOE_BUILD_SHARED=1
CPPFLAGS += -DWEFTMOE_DEVICE_TIMELINE=$(WEFTMOE_DEVICE_TIMELINE)
CUDA_CPPFLAGS += -DWEFTMOE_DEVICE_TIMELINE=$(WEFTMOE_DEVICE_TIMELINE)
CPPFLAGS += -DWEFTMOE_FORWARD_LIGHT_PROGRESS=$(WEFTMOE_FORWARD_LIGHT_PROGRESS)
CUDA_CPPFLAGS += -DWEFTMOE_FORWARD_LIGHT_PROGRESS=$(WEFTMOE_FORWARD_LIGHT_PROGRESS)
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

# The persistent WGMMA/TMA kernels require Hopper architecture features.
CUDA_ARCH_FLAGS ?= -gencode arch=compute_90a,code=sm_90a
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
CUDA_LIB_SOURCES := \
	src/cuda/device_notify.cu \
	src/cuda/activation_transport.cu \
	src/cuda/forward_transport.cu \
	src/cuda/persistent_v3/kernel_entry.cu \
	src/transport/nccl_count_exchange.cu
CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/%.o)
PROD_CUDA_LIB_OBJECTS := \
	$(CUDA_LIB_SOURCES:%.cu=$(BUILD_DIR)/production/%.o)

.PHONY: all test launcher-security-test forward-autotune-test forward-autotune backward-autotune-test backward-autotune header-test api-negative-test shared-lib testing-lib install uninstall-manifest install-test installed-header-scan release-text-check public-symbol-check retired-symbol-link-negative kernel-artifact-check kernel-artifact-update license-check release-test release-check release-check-inner bench megatron-bench megatron-backward-bench megatron-flex-bench megatron-flex-stage-profile megatron-flex-backward-bench fair-integrated-bench cuda-lib cuda-test cuda-test-local cuda-protocol-test cuda-bench cuda-transport-test cuda-p2p-test cuda-p2p-bench cuda-integrated-bench cuda-transport-bench cuda-sanitizer-smoke cuda-sm90-compile clean

FWD_AUTOTUNE_CONFIG ?=
BWD_AUTOTUNE_CONFIG ?=
AUTOTUNE_ARGS ?=

forward-autotune-test:
	python3 tests/test_forward_autotune.py

forward-autotune:
	python3 tools/autotune_forward.py $(if $(FWD_AUTOTUNE_CONFIG),--config $(FWD_AUTOTUNE_CONFIG)) $(AUTOTUNE_ARGS)

backward-autotune-test:
	python3 tests/test_backward_autotune.py

backward-autotune:
	python3 tools/autotune_backward.py $(if $(BWD_AUTOTUNE_CONFIG),--config $(BWD_AUTOTUNE_CONFIG)) $(AUTOTUNE_ARGS)

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/metadata_dump $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -DWEFTMOE_BUILD_TESTING=1 $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DWEFTMOE_BUILD_TESTING=1 $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -MMD -MP -c $< -o $@

$(BUILD_DIR)/production/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -DWEFTMOE_BUILD_TESTING=0 $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/production/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CPPFLAGS) -DWEFTMOE_BUILD_TESTING=0 $(CUDA_NVCCFLAGS) $(CUDA_ARCH_FLAGS) -Xcompiler=-Wall,-Wextra,-Werror -MMD -MP -c $< -o $@

# Track actual transitive headers instead of maintaining fragment lists.
-include $(LIB_OBJECTS:.o=.d) $(PROD_LIB_OBJECTS:.o=.d) \
         $(CUDA_LIB_OBJECTS:.o=.d) $(PROD_CUDA_LIB_OBJECTS:.o=.d)

# Transport intentionally leaves compute/host helpers uninstantiated.
$(BUILD_DIR)/src/cuda/forward_transport.o \
$(BUILD_DIR)/production/src/cuda/forward_transport.o: CUDA_NVCCFLAGS += -diag-suppress=177

cuda-lib: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)

$(BUILD_DIR)/$(LIBWEFTMOE_REALNAME): $(PROD_LIB_OBJECTS) $(PROD_CUDA_LIB_OBJECTS) abi/libweftmoe.map
	@mkdir -p $(@D)
	$(NVCC) -Wno-deprecated-gpu-targets -shared \
		-Xlinker=-soname -Xlinker=$(LIBWEFTMOE_SONAME) \
		-Xlinker=--version-script=$(CURDIR)/abi/libweftmoe.map \
		-Xlinker=--gc-sections \
		$(LDFLAGS) $(filter %.o,$^) -lnccl -lcublas -lcudart -lcuda -ldl -o $@

shared-lib: $(BUILD_DIR)/$(LIBWEFTMOE_REALNAME)

$(BUILD_DIR)/libweftmoe_testing.a: $(LIB_OBJECTS) $(CUDA_LIB_OBJECTS)
	@mkdir -p $(@D)
	ar rcs $@ $^

# Repository tests may link this non-installed archive. Production packages
# only build the WEFTMOE_BUILD_TESTING=0 object graph above.
testing-lib: $(BUILD_DIR)/libweftmoe_testing.a

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
	python3 tools/check_public_symbols.py --binary $(BUILD_DIR)/$(LIBWEFTMOE_REALNAME) \
		--whitelist abi/public_symbols_v2.txt

retired-symbol-link-negative: shared-lib
	ln -sfn $(LIBWEFTMOE_REALNAME) $(BUILD_DIR)/libweftmoe.so
	python3 tools/check_retired_symbol_link.py --library-dir $(abspath $(BUILD_DIR))

license-check:
	test -s LICENSE
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
$(BUILD_DIR)/test_cuda_notify: $(BUILD_DIR)/libweftmoe_testing.a $(CUDA_TEST_SOURCES)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_ARCH_FLAGS) $(CUDA_NVCCFLAGS) $(CUDA_CPPFLAGS) $(LDFLAGS) \
		$(CUDA_TEST_SOURCES) -Xlinker=--whole-archive \
		$(BUILD_DIR)/libweftmoe_testing.a -Xlinker=--no-whole-archive \
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

megatron-flex-bench:
	torchrun --standalone --nproc-per-node=8 \
		benchmarks/megatron_dispatch_baseline.py --dispatcher flex \
		--deepep-num-sms 64 --expert-body te-grouped-mlp

megatron-flex-stage-profile:
	torchrun --standalone --nproc-per-node=8 \
		benchmarks/megatron_dispatch_baseline.py --dispatcher flex \
		--deepep-num-sms 64 --expert-body te-grouped-mlp \
		--stage-profile --runs 1

megatron-flex-backward-bench:
	torchrun --standalone --nproc-per-node=8 \
		benchmarks/megatron_backward_baseline.py --dispatcher flex \
		--deepep-num-sms 64

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
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) \
		$(LIB_OBJECTS) tests/test_notify_dispatch.cpp -o $@

$(BUILD_DIR)/notify_dispatch_bench: $(LIB_OBJECTS) benchmarks/notify_dispatch_bench.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/metadata_dump: $(LIB_OBJECTS) tests/metadata_dump.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

launcher-security-test:
	PYTHONPATH=. python3 tests/test_launcher_security.py

header-test:
	CXX='$(CXX)' python3 tests/check_headers.py

api-negative-test:
	CXX='$(CXX)' python3 tests/check_cuda_public_api.py

install: shared-lib
	@for header in $(PUBLIC_HEADERS); do \
		destination='$(DESTDIR)$(INCLUDEDIR)/'$${header#include/}; \
		install -D -m 0644 $$header $$destination; \
	done
	install -D -m 0755 $(BUILD_DIR)/$(LIBWEFTMOE_REALNAME) \
		$(DESTDIR)$(LIBDIR)/$(LIBWEFTMOE_REALNAME)
	ln -sfn $(LIBWEFTMOE_REALNAME) $(DESTDIR)$(LIBDIR)/$(LIBWEFTMOE_SONAME)
	ln -sfn $(LIBWEFTMOE_SONAME) $(DESTDIR)$(LIBDIR)/libweftmoe.so
	@mkdir -p $(DESTDIR)$(PKGCONFIGDIR)
	printf '%s\n' 'prefix=$(PREFIX)' 'exec_prefix=$${prefix}' 'libdir=$${exec_prefix}/lib' 'includedir=$${prefix}/include' '' 'Name: WeftMoE' 'Description: Deterministic persistent MoE runtime' 'Version: $(LIBWEFTMOE_VERSION)' 'Libs: -L$${libdir} -lweftmoe' 'Libs.private: -lnccl -lcudart -lcuda -ldl' 'Cflags: -I$${includedir}' > $(DESTDIR)$(PKGCONFIGDIR)/weftmoe.pc
	install -D -m 0644 examples/minimal_forward.cpp \
		$(DESTDIR)$(DATADIR)/examples/minimal_forward.cpp
	install -D -m 0644 LICENSE $(DESTDIR)$(DATADIR)/LICENSE
	install -D -m 0644 LICENSES/DeepGEMM-MIT.txt $(DESTDIR)$(DATADIR)/LICENSES/DeepGEMM-MIT.txt
	install -D -m 0644 THIRD_PARTY_NOTICES.md \
		$(DESTDIR)$(DATADIR)/THIRD_PARTY_NOTICES.md
	@{ \
		for header in $(PUBLIC_HEADERS); do echo '$(INCLUDEDIR)/'$${header#include/}; done; \
		echo '$(LIBDIR)/$(LIBWEFTMOE_REALNAME)'; \
		echo '$(LIBDIR)/$(LIBWEFTMOE_SONAME)'; \
		echo '$(LIBDIR)/libweftmoe.so'; \
		echo '$(PKGCONFIGDIR)/weftmoe.pc'; \
		echo '$(DATADIR)/examples/minimal_forward.cpp'; \
		echo '$(DATADIR)/LICENSE'; \
		echo '$(DATADIR)/LICENSES/DeepGEMM-MIT.txt'; \
		echo '$(DATADIR)/THIRD_PARTY_NOTICES.md'; \
		echo '$(DATADIR)/install_manifest.txt'; \
	} > $(DESTDIR)$(DATADIR)/install_manifest.txt

uninstall-manifest:
	@printf '%s\n' 'Installed files are listed in $(DATADIR)/install_manifest.txt.' \
		'This target is intentionally non-destructive; remove exactly that manifest.'

install-test:
	CXX='$(CXX)' python3 tests/check_install.py

test: $(BUILD_DIR)/test_notify_dispatch launcher-security-test header-test api-negative-test
	$(BUILD_DIR)/test_notify_dispatch

ifeq ($(ENABLE_CUDA),1)
test: cuda-test
endif

bench: $(BUILD_DIR)/notify_dispatch_bench
	$(BUILD_DIR)/notify_dispatch_bench

clean:
	rm -rf -- $(BUILD_DIR)
