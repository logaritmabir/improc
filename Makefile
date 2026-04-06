# Makefile for improc

EXECUTABLE_NAME = improc
TEST_NAME = test

INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = .obj
BIN_DIR = bin
TEST_DIR = tests

CUDA_SRC_DIR = $(SRC_DIR)/cuda
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda

CC = g++
NVCC = nvcc
CC_FLAGS = -I$(INCLUDE_DIR) -std=c++11 -g
NVCC_FLAGS = -I$(INCLUDE_DIR) -O2 -arch=sm_75
NVCC_LINK_FLAGS = -lcudart
GTEST_FLAGS = -Llib -lgtest -lgtest_main -pthread

# Build flag: set BUILD_WITH_CUDA=1 to enable CUDA support
BUILD_WITH_CUDA ?= 0

CPP_FILES = $(shell find $(SRC_DIR) -name "*.cpp")
MAIN_FILE = $(SRC_DIR)/main.cpp
CPP_FILES_NO_MAIN = $(filter-out $(MAIN_FILE), $(CPP_FILES))

ifeq ($(BUILD_WITH_CUDA),1)
CUDA_FILES = $(shell find $(CUDA_SRC_DIR) -name "*.cu")
CUDA_OBJ_FILES = $(patsubst $(CUDA_SRC_DIR)/%.cu, $(CUDA_OBJ_DIR)/%.o, $(CUDA_FILES))
CC_FLAGS += -DBUILD_WITH_CUDA
else
CUDA_FILES =
CUDA_OBJ_FILES =
endif

TEST_FILES = $(shell find $(TEST_DIR) -name "*_test.cpp")

CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_FILES_NO_MAIN))
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/tests/%.o, $(TEST_FILES))

default: all

all: $(BIN_DIR)/$(EXECUTABLE_NAME)

ifeq ($(BUILD_WITH_CUDA),1)
$(BIN_DIR)/$(EXECUTABLE_NAME): $(CPP_OBJ_FILES) $(CUDA_OBJ_FILES) $(OBJ_DIR)/main.o | $(BIN_DIR)
	$(NVCC) $(CC_FLAGS) $^ -o $@ $(NVCC_LINK_FLAGS)
else
$(BIN_DIR)/$(EXECUTABLE_NAME): $(CPP_OBJ_FILES) $(OBJ_DIR)/main.o | $(BIN_DIR)
	$(CC) $(CC_FLAGS) $^ -o $@
endif

test: $(BIN_DIR)/$(TEST_NAME)

ifeq ($(BUILD_WITH_CUDA),1)
$(BIN_DIR)/$(TEST_NAME): $(TEST_OBJ_FILES) $(CPP_OBJ_FILES) $(CUDA_OBJ_FILES) | $(BIN_DIR)
	$(NVCC) $(CC_FLAGS) $^ -o $@ $(NVCC_LINK_FLAGS) $(GTEST_FLAGS)
else
$(BIN_DIR)/$(TEST_NAME): $(TEST_OBJ_FILES) $(CPP_OBJ_FILES) | $(BIN_DIR)
	$(CC) $(CC_FLAGS) $^ -o $@ $(GTEST_FLAGS)
endif

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
ifeq ($(BUILD_WITH_CUDA),1)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
else
	$(CC) $(CC_FLAGS) -c $< -o $@
endif

$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)/tests
	$(CC) $(CC_FLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $@

$(OBJ_DIR):
	mkdir -p $@

$(CUDA_OBJ_DIR): | $(OBJ_DIR)
	mkdir -p $@

$(OBJ_DIR)/tests: | $(OBJ_DIR)
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
