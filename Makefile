# Makefile for improc

# Build type: set BUILD_TYPE=release for release build, default is debug
BUILD_TYPE ?= debug
# Build flag: set BUILD_WITH_CUDA=1 to enable CUDA support
BUILD_WITH_CUDA ?= 0

TARGET_NAME = improc
TEST_TARGET_NAME = test

INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = .obj
BIN_DIR = bin
TEST_DIR = tests
INSTALL_DIR ?= install

CUDA_SRC_DIR = $(SRC_DIR)/cuda
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda

CXX = g++
NVCC = nvcc

CXXFLAGS = -I$(INCLUDE_DIR) -std=c++11 -Ilib/include -MMD -MP
NVCCFLAGS = -I$(INCLUDE_DIR) --std=c++11 -O2 -arch=sm_75 -lineinfo -Xcompiler -MMD -MP

ifneq ($(BUILD_TYPE),release)
	CXXFLAGS += -O0 -g -Wall -Wextra -Wpedantic -Werror
else
	CXXFLAGS += -O2 -DNDEBUG
endif

CUDA_LDLIBS = -lcudart -lnppif
GTEST_LDLIBS = -Llib -lgtest -lgtest_main -pthread

CXX_SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
CXX_MAIN_SRC = $(SRC_DIR)/main.cpp
CXX_SRCS_NO_MAIN = $(filter-out $(CXX_MAIN_SRC), $(CXX_SRCS))
CXX_TEST_SRCS = $(shell find $(TEST_DIR) -name "*_test.cpp")

CXX_OBJS_NO_MAIN = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CXX_SRCS_NO_MAIN))
CXX_MAIN_OBJ = $(OBJ_DIR)/main.o
CXX_TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/tests/%.o, $(CXX_TEST_SRCS))

ifeq ($(BUILD_WITH_CUDA),1)
CUDA_SRCS = $(shell find $(CUDA_SRC_DIR) -name "*.cu")
CUDA_OBJS = $(patsubst $(CUDA_SRC_DIR)/%.cu, $(CUDA_OBJ_DIR)/%.o, $(CUDA_SRCS))
CXXFLAGS += -DBUILD_WITH_CUDA
NVCCFLAGS += -DBUILD_WITH_CUDA
else
CUDA_SRCS =
CUDA_OBJS =
endif

ALL_DEPS = $(CXX_OBJS_NO_MAIN:.o=.d) $(CXX_MAIN_OBJ:.o=.d) $(CXX_TEST_OBJS:.o=.d) $(CUDA_OBJS:.o=.d)

.PHONY: default all test static_lib static_analysis install help clean

default: all

all: $(BIN_DIR)/$(TARGET_NAME)

ifeq ($(BUILD_WITH_CUDA),1)
$(BIN_DIR)/$(TARGET_NAME): $(CXX_OBJS_NO_MAIN) $(CUDA_OBJS) $(CXX_MAIN_OBJ) | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $^ -o $@ $(CUDA_LDLIBS)
else
$(BIN_DIR)/$(TARGET_NAME): $(CXX_OBJS_NO_MAIN) $(CXX_MAIN_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@
endif

test: $(BIN_DIR)/$(TEST_TARGET_NAME)

ifeq ($(BUILD_WITH_CUDA),1)
$(BIN_DIR)/$(TEST_TARGET_NAME): $(CXX_TEST_OBJS) $(CXX_OBJS_NO_MAIN) $(CUDA_OBJS) | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $^ -o $@ $(CUDA_LDLIBS) $(GTEST_LDLIBS)
else
$(BIN_DIR)/$(TEST_TARGET_NAME): $(CXX_TEST_OBJS) $(CXX_OBJS_NO_MAIN) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(GTEST_LDLIBS)
endif

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	mkdir -p $(dir $@)
ifeq ($(BUILD_WITH_CUDA),1)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $@

$(INSTALL_DIR): static_lib
	mkdir -p $@
	mkdir -p $@/include
	mkdir -p $@/lib
	cp $(BIN_DIR)/lib$(TARGET_NAME).a $@/lib
	cp -r $(INCLUDE_DIR)/* $@/include

static_lib: $(CXX_OBJS_NO_MAIN) | $(BIN_DIR)
	ar rcs $(BIN_DIR)/lib$(TARGET_NAME).a $^

static_analysis:
	cppcheck --enable=all --error-exitcode=1 --inconclusive --std=c++11 --suppress=missingIncludeSystem -I$(INCLUDE_DIR) -Ilib/include $(SRC_DIR)

help:
	@echo "Targets:"
	@echo "  all              Build the improc executable (default)"
	@echo "  test             Build the test binary"
	@echo "  static_lib       Build static library libimproc.a"
	@echo "  static_analysis  Run cppcheck on src/ and tests/"
	@echo "  clean            Remove build outputs"
	@echo "  help             Show this message"

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(INSTALL_DIR)

-include $(ALL_DEPS)