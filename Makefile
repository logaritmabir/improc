#Makefile

EXECUTABLE_NAME = improc
TEST_NAME = test

INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = .obj
BIN_DIR = bin
TEST_DIR = tests

CC = g++
CC_FLAGS = -I$(INCLUDE_DIR) -std=c++11 -g
GTEST_FLAGS = -lgtest -lgtest_main -pthread

SRC_FILES = $(filter-out $(SRC_DIR)/main.cpp, $(shell find $(SRC_DIR) -name "*.cpp"))
TEST_FILES = $(shell find $(TEST_DIR) -name "*_test.cpp")
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))

default : all

all: $(BIN_DIR)/$(EXECUTABLE_NAME)

$(BIN_DIR)/$(EXECUTABLE_NAME): $(OBJ_FILES) $(SRC_DIR)/main.cpp
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi
	$(CC) $(CC_FLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	$(CC) $(CC_FLAGS) -c $< -o $@

test : $(BIN_DIR)/$(TEST_NAME)

$(BIN_DIR)/$(TEST_NAME): $(TEST_FILES) $(OBJ_FILES)
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi
	$(CC) $(CC_FLAGS) $^ -o $@ $(GTEST_FLAGS)

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	$(CC) $(CC_FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/$(EXECUTABLE_NAME)
