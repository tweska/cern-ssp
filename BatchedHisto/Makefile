.PHONY: all clean

ROOT_FLAGS      = $(shell root-config --cflags)
ROOT_LIBS       = $(shell root-config --libs)

CPP_COMPILER    = g++
CPP_FLAGS       = -O3 -g -Wall -Werror -I../Common $(ROOT_FLAGS)
CPP_LIBS        = $(ROOT_LIBS)

CUDA_COMPILER   = nvcc
CUDA_FLAGS      = -arch=sm_75 -g -G -I../Common
CUDA_LIBS       = -lcuda -lcudart

TEST_LIBS       = -lgtest -lgtest_main

TARGET          = bin/GbHistoTest

all: $(TARGET)

obj/%.cpp.o: %.cpp
	@ mkdir -p obj
	$(CPP_COMPILER) -c $< -o $@ $(CPP_FLAGS)

obj/%.cu.o: %.cu
	@ mkdir -p obj
	$(CUDA_COMPILER) -c $< -o $@ $(CUDA_FLAGS)

bin/GbHistoTest: obj/GbHisto.cu.o obj/GbHistoTest.cpp.o
	@ mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS) $(CUDA_LIBS) $(TEST_LIBS)

clean:
	rm -rf bin/ obj/
