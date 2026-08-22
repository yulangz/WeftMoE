CXX ?= c++
BUILD_DIR ?= build
CPPFLAGS := -Iinclude
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow

LIB_SOURCES := \
	src/metadata/notify_dispatch.cpp \
	src/transport/in_memory_count_exchange.cpp \
	src/scoreboard/workspace_plan.cpp
LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all test reference-test bench clean

all: $(BUILD_DIR)/test_notify_dispatch $(BUILD_DIR)/notify_dispatch_bench

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_notify_dispatch: $(LIB_OBJECTS) tests/test_notify_dispatch.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/notify_dispatch_bench: $(LIB_OBJECTS) benchmarks/notify_dispatch_bench.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

reference-test:
	PYTHONPATH=. python3 -m unittest discover -v -s tests -p 'test_reference.py'

test: $(BUILD_DIR)/test_notify_dispatch reference-test
	$(BUILD_DIR)/test_notify_dispatch

bench: $(BUILD_DIR)/notify_dispatch_bench
	$(BUILD_DIR)/notify_dispatch_bench

clean:
	rm -rf -- $(BUILD_DIR)
