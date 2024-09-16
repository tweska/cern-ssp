.PHONY: all clean

ROOT_FLAGS    = $(shell root-config --cflags)
ROOT_LIBS     = $(shell root-config --libs)

CPP_COMPILER  = g++
CPP_FLAGS     = -O3 -g -Wall -Werror -Iinc -I../Common $(ROOT_FLAGS)
CPP_LIBS      = $(ROOT_LIBS)

CUDA_COMPILER = nvcc
CUDA_FLAGS    = -O3 -arch=sm_75 -g -G -Iinc -I../Common
CUDA_LIBS     = -lcuda -lcudart

TARGETS       = bin/CpuFWMOriginal bin/CpuFWM bin/GpuFWM

ifdef UNSTABLE_INVARIANT_MASS
	CPP_FLAGS += -DUNSTABLE_INVARIANT_MASS
	CUDA_FLAGS += -DUNSTABLE_INVARIANT_MASS
endif

all: $(TARGETS)

obj/%.cpp.o: src/%.cpp
	@mkdir -p obj
	$(CPP_COMPILER) -c $< -o $@ $(CPP_FLAGS)

obj/%.cu.o: src/%.cu
	@mkdir -p obj
	$(CUDA_COMPILER) -c $< -o $@ $(CUDA_FLAGS)

bin/CpuFWMOriginal: CpuFWMOriginal.cpp obj/CpuFWM.cpp.o
	@mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS)

bin/CpuFWM: CpuFWM.cpp obj/CpuFWM.cpp.o obj/coords.cpp.o
	@mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS)

bin/GpuFWM: GpuFWM.cpp obj/GpuFWM.cu.o obj/coords.cpp.o
	@mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS) $(CUDA_LIBS)

clean:
	rm -rf bin/ obj/
