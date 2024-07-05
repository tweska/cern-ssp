#ROOT_INCLUDES = $(patsubst %,-I %,$(wildcard root/*/inc) $(wildcard root/*/*/inc))

CPP_COMPILER  = g++
CPP_FLAGS     = -std=c++11 -Wall -Wextra -O3 -march=native -Wall -Wextra
CPP_INCLUDES  = -I inc $(ROOT_INCLUDES)
CPP_LIBS      = -L/usr/local/cuda/lib64 -lcuda -lcudart

CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75
CUDA_INCLUDES = $(CPP_INCLUDES)

SOURCE_FILES  = $(filter-out src/RHnCUDA-impl.cu, $(wildcard src/*.cu)) $(wildcard src/*.cpp)
OBJECT_FILES  = $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(SOURCE_FILES)))


all: main

%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

#%.cpp.o: %.cpp
#	$(CPP_COMPILER) $(CPP_FLAGS) $(CPP_INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

main: src/main.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CPP_LIBS) $(CUDA_INCLUDES) $^ -o $@

clean:
	rm -f main src/*.o

.PHONY: all clean
