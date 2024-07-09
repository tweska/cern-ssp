CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75 -G
CUDA_INCLUDES = -I inc $(ROOT_INCLUDES)

SOURCE_FILES  = $(filter-out src/RHnCUDA-impl.cu, $(wildcard src/*.cu)) $(wildcard src/*.cpp)
OBJECT_FILES  = $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(SOURCE_FILES)))


all: main

%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

main: $(OBJECT_FILES)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CPP_LIBS) $(CUDA_INCLUDES) $^ -o $@

clean:
	rm -f main src/*.o

.PHONY: all clean
