ROOT_FLAGS    = $(shell root-config --cflags)
ROOT_LIBS     = $(shell root-config --libs)

CPP_COMPILER  = g++
CPP_FLAGS     = -O3 -g -Wall -Werror $(ROOT_FLAGS)
CPP_LIBS      = $(ROOT_LIBS)

CUDA_COMPILER = nvcc
CUDA_FLAGS    = -arch=sm_75 -g -G
CUDA_LIBS     = -lcuda -lcudart

SOURCE_FILES  = $(wildcard *.cpp) $(wildcard *.cu)
OBJECT_FILES  = $(patsubst %.cpp, %.o, $(patsubst %.cu, %.o, $(SOURCE_FILES)))


all: main

%.o: %.cpp
	$(CPP_COMPILER) -c $< -o $@ $(CPP_FLAGS)

%.o: %.cu
	$(CUDA_COMPILER) -c $< -o $@ $(CUDA_FLAGS)

main: $(OBJECT_FILES)
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS) $(CUDA_LIBS)

clean:
	rm -rf main *.o

.PHONY: all clean
