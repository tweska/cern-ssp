ROOT_INCLUDES = -I/snap/root-framework/931/usr/local/include
ROOT_LIBS     = -L/snap/root-framework/931/usr/local/lib -lCore -lHist

CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75 -G
CUDA_INCLUDES = -Iinc

SOURCE_FILES  = $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(SOURCE_FILES)))


all: main

%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(ROOT_INCLUDES) -c $< -o $@

main: $(OBJECT_FILES)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(ROOT_INCLUDES) $(ROOT_LIBS) $^ -o $@

clean:
	rm -f main src/*.o

.PHONY: all clean
