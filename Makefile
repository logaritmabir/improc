#Makefile

EXECUTABLE_NAME = improc
TEST_NAME = test

INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = .obj
BIN_DIR = bin
TEST_DIR = tests

CC = g++
NVCC = nvcc
CC_FLAGS = -I$(INCLUDE_DIR) -std=c++11 -g
NVCC_FLAGS = -I$(INCLUDE_DIR) -O2 -arch=sm_75
NVCC_LINK_FLAGS = -lcudart
GTEST_FLAGS = -lgtest -lgtest_main -pthread

CPP_FILES = $(filter-out $(SRC_DIR)/main.cpp, $(shell find $(SRC_DIR) -name "*.cpp"))
CUDA_FILES = $(shell find $(SRC_DIR)/cuda -name "*.cu")
TEST_FILES = $(shell find $(TEST_DIR) -name "*_test.cpp")

CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_FILES))
CUDA_OBJ_FILES = $(patsubst $(SRC_DIR)/cuda/%.cu, $(OBJ_DIR)/%.o, $(CUDA_FILES))

default : all

all: $(BIN_DIR)/$(EXECUTABLE_NAME)

$(BIN_DIR)/$(EXECUTABLE_NAME): $(CPP_OBJ_FILES) $(CUDA_OBJ_FILES) $(SRC_DIR)/main.cpp
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi
	$(NVCC) $(CC_FLAGS) $^ -o $@ $(NVCC_LINK_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/cuda/%.cu
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

test : $(BIN_DIR)/$(TEST_NAME)

$(BIN_DIR)/$(TEST_NAME): $(TEST_FILES) $(CPP_OBJ_FILES)
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi
	$(CC) $(CC_FLAGS) $^ -o $@ $(GTEST_FLAGS)

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	$(CC) $(CC_FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/$(EXECUTABLE_NAME)
