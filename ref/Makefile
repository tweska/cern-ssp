ROOT_INCLUDES = -I$(shell root-config --incdir)
ROOT_LIBS     = -Xlinker -rpath -Xlinker $(shell root-config --libdir) -L$(shell root-config --libdir) -lCore -lHist

CUDA_COMPILER = /usr/local/cuda-12.5/bin/nvcc
CUDA_FLAGS    = -arch=sm_75 -G
CUDA_INCLUDES = -Iinc $(ROOT_INCLUDES)
CUDA_LIBS     = $(ROOT_LIBS)

SOURCE_FILES  = $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst src/%.cu, obj/%.o, $(SOURCE_FILES))


all: bin/main

obj/%.o: src/%.cu
	@ mkdir -p obj
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@

bin/main: $(OBJECT_FILES)
	@ mkdir -p bin
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(CUDA_LIBS) $^ -o $@

clean:
	rm -rf bin/ obj/

.PHONY: all clean
