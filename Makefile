ROOT_INCLUDES = -I$(shell root-config --incdir)
ROOT_LIBS     = -Xlinker -rpath -Xlinker $(shell root-config --libdir) -L$(shell root-config --libdir) -lCore -lHist

CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75 -G
CUDA_INCLUDES = -Iinc $(ROOT_INCLUDES)
CUDA_LIBS     = $(ROOT_LIBS)

SOURCE_FILES  = $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst %.cu, %.o, $(SOURCE_FILES))


all: main

%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

main: $(OBJECT_FILES)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(CUDA_LIBS) $^ -o $@

clean:
	rm -f main src/*.o

.PHONY: all clean
