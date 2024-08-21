ROOT_FLAGS      = $(shell root-config --cflags)
ROOT_LIBS       = $(shell root-config --libs)

CPP_COMPILER    = g++
CPP_FLAGS       = -O3 -g -Wall -Werror -Iinc $(ROOT_FLAGS)
CPP_LIBS        = $(ROOT_LIBS)

CUDA_COMPILER   = nvcc
CUDA_FLAGS      = -arch=sm_75 -g -G -Iinc
CUDA_LIBS       = -lcuda -lcudart

TEST_LIBS       = -lgtest -lgtest_main

SOURCE_FILES    = $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJECT_FILES    = $(patsubst src/%.cpp, obj/%.cpp.o, $(patsubst src/%.cu, obj/%.cu.o, $(SOURCE_FILES)))

all: bin/test

obj/%.cpp.o: src/%.cpp
	@ mkdir -p obj
	$(CPP_COMPILER) -c $< -o $@ $(CPP_FLAGS)

obj/%.cu.o: src/%.cu
	@ mkdir -p obj
	$(CUDA_COMPILER) -c $< -o $@ $(CUDA_FLAGS)

bin/test: $(OBJECT_FILES)
	@ mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS) $(CUDA_LIBS) $(TEST_LIBS)

clean:
	rm -rf bin/ obj/

.PHONY: all clean
