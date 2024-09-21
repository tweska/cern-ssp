#!/bin/bash

make clean && make all

cd BatchedHisto
echo "BatchedHisto"
echo "CPU"
bin/CpuPerf
echo "GPU"
bin/GpuPerf
cd ../

cd DiMuon
echo "DiMuon"
echo "CPU"
bin/DiMuonCpuPerf
echo "GPU"
bin/DiMuonGpuPerf
cd ../

cd FoldedWmass
echo "FoldedWmass"
echo "CPU Stable"
bin/CpuFWM 10 --warmup
echo "GPU Stable"
bin/GpuFWM 10 --warmup
make -B all UNSTABLE_INVARIANT_MASS=1
echo "CPU Unstable"
bin/CpuFWM 10 --warmup
echo "GPU Unstable"
bin/GpuFWM 10 --warmup
make -B all
cd ../
