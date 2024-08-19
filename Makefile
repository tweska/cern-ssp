ROOT_CPP_FLAGS = $(shell root-config --cflags --libs)
ROOT_NVCC_FLAGS = -I$(shell root-config --incdir) -Xlinker -rpath -Xlinker $(shell root-config --libdir) -L$(shell root-config --libdir) -lCore -lHist


CPP_COMPILER  = g++
CPP_FLAGS     = -O3 -g -Wall -Werror -Iinc $(ROOT_CPP_FLAGS)

CUDA_COMPILER = nvcc
CUDA_FLAGS    = -arch=sm_75 -g -G -Iinc $(ROOT_NVCC_FLAGS)

TEST_LIBS     = -lgtest -lgtest_main

SOURCE_FILES  = $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst src/%.cpp, obj/%.cpp.o, $(patsubst src/%.cu, obj/%.cu.o, $(SOURCE_FILES)))

EXPERIMENT_SOURCE_FILES = $(wildcard experiment/*.cpp) $(wildcard experiment/*.cu)
EXPERIMENT_TARGET_FILES = $(patsubst %.cpp, %, $(patsubst %.cu, %, $(EXPERIMENT_SOURCE_FILES)))

all: bin/test $(EXPERIMENT_TARGET_FILES)

obj/%.cpp.o: src/%.cpp
	@ mkdir -p obj
	$(CPP_COMPILER) $(CPP_FLAGS) -c $< -o $@

obj/%.cu.o: src/%.cu
	@ mkdir -p obj
	$(CUDA_COMPILER) $(CUDA_FLAGS) -c $< -o $@

bin/test: $(OBJECT_FILES)
	@ mkdir -p bin
	$(CUDA_COMPILER) -o $@ $^ $(CUDA_FLAGS) $(TEST_LIBS)

experiment/%: experiment/%.cpp
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS)

experiment/%: experiment/%.cu
	$(CUDA_COMPILER) -o $@ $^ $(CUDA_FLAGS)

clean:
	rm -rf bin/ obj/ $(EXPERIMENT_TARGET_FILES)

.PHONY: all clean
