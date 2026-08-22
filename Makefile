CXX ?= c++
BUILD_DIR ?= build
CPPFLAGS := -Iinclude -Isrc -pthread
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow

LIB_SOURCES := \
	src/metadata/notify_dispatch.cpp \
	src/metadata/serialize.cpp \
	src/transport/host_fake_count_exchange.cpp \
	src/scoreboard/workspace_plan.cpp
LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all test reference-test cross-check header-test bench clean

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/metadata_dump $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_notify_dispatch: $(LIB_OBJECTS) tests/test_notify_dispatch.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/notify_dispatch_bench: $(LIB_OBJECTS) benchmarks/notify_dispatch_bench.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/metadata_dump: $(LIB_OBJECTS) tests/metadata_dump.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

reference-test:
	PYTHONPATH=. python3 -m unittest discover -v -s tests -p 'test_reference.py'

cross-check: $(BUILD_DIR)/metadata_dump
	$(BUILD_DIR)/metadata_dump > $(BUILD_DIR)/cpp_metadata.json
	PYTHONPATH=. python3 tests/compare_reference.py $(BUILD_DIR)/cpp_metadata.json

header-test:
	CXX='$(CXX)' python3 tests/check_headers.py

test: $(BUILD_DIR)/test_notify_dispatch reference-test cross-check header-test
	$(BUILD_DIR)/test_notify_dispatch

bench: $(BUILD_DIR)/notify_dispatch_bench
	$(BUILD_DIR)/notify_dispatch_bench

clean:
	rm -rf -- $(BUILD_DIR)
