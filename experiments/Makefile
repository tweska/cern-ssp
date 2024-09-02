ROOT_FLAGS    = $(shell root-config --cflags)
ROOT_LIBS     = $(shell root-config --libs)

CPP_COMPILER  = g++
CPP_FLAGS     = -O3 -g -Wall -Werror -Iinc -I../inc $(ROOT_FLAGS)
CPP_LIBS      = $(ROOT_LIBS)

CUDA_COMPILER = nvcc
CUDA_FLAGS    = -arch=sm_75 -g -G -Iinc -I../inc
CUDA_LIBS     = -lcuda -lcudart

SOURCE_FILES  = $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJECT_FILES  = $(patsubst src/%.cpp, obj/%.cpp.o, $(patsubst src/%.cu, obj/%.cu.o, $(SOURCE_FILES)))

DATA          = data/Run2012BC_DoubleMuParked_Muons.root
TARGETS       = bin/DiMuonMaxError bin/DiMuonBinCheck bin/DiMuonCpuPerf bin/DiMuonGpuPerf

all: $(DATA) $(TARGETS)

obj/%.cpp.o: src/%.cpp
	@mkdir -p obj
	$(CPP_COMPILER) -c $< -o $@ $(CPP_FLAGS)

obj/%.cu.o: src/%.cu
	@mkdir -p obj
	$(CUDA_COMPILER) -c $< -o $@ $(CUDA_FLAGS)

bin/%: %.cpp $(OBJECT_FILES)
	@mkdir -p bin
	$(CPP_COMPILER) -o $@ $^ $(CPP_FLAGS) $(CPP_LIBS) $(CUDA_LIBS)

data/Run2012BC_DoubleMuParked_Muons.root:
	@mkdir -p data
	wget https://opendata.web.cern.ch/record/12341/files/Run2012BC_DoubleMuParked_Muons.root -O data/Run2012BC_DoubleMuParked_Muons.root

clean:
	rm -rf bin/ obj/

.PHONY: all clean
.PRECIOUS: $(OBJECT_FILES)
