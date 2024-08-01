ROOT_INCLUDES = -I$(shell root-config --incdir)
ROOT_LIBS     = -Xlinker -rpath -Xlinker $(shell root-config --libdir) -L$(shell root-config --libdir) -lCore -lHist

CPP_COMPILER  = g++
CPP_FLAGS     = -O3
CPP_INCLUDES  = -Iinc $(ROOT_INCLUDES)
CPP_LIBS      = $(ROOT_LIBS)

CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75 -G
CUDA_INCLUDES = -Iinc $(ROOT_INCLUDES)
CUDA_LIBS     = $(ROOT_LIBS)

SOURCE_FILES  = $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst src/%.cpp, obj/%.cpp.o, $(patsubst src/%.cu, obj/%.cu.o, $(SOURCE_FILES)))


all: bin/main

obj/%.cpp.o: src/%.cpp
	@ mkdir -p obj
	$(CPP_COMPILER) $(CPP_FLAGS) $(CPP_INCLUDES) -c $< -o $@

obj/%.cu.o: src/%.cu
	@ mkdir -p obj
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

bin/main: $(OBJECT_FILES)
	@ mkdir -p bin
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(CUDA_LIBS) $^ -o $@

clean:
	rm -rf bin/ obj/

.PHONY: all clean
