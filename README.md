# ROOT GPU Optimization Experiments for High Energy Physics Analysis


## Dependencies and Compilers
| Package                               | Version                                                                                                  | Notes             |
|---------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------|
| ROOT                                  | 6.33.01 ([5b7b6cc0](https://github.com/root-project/root/tree/5b7b6cc0253a145ee59b8890b6e26d81ac7198b4)) | Build from source |
| Google Test                           | 1.15.2 ([b514bdc8](https://github.com/google/googletest/tree/b514bdc898e2951020cbdca1304b75f5950d1f59))  | Build from source |
| GNU Compiler Collection (`gcc`/`g++`) | 12.3.0                                                                                                   |                   |
| CUDA Toolkit (includes `nvcc`)        | 12.5                                                                                                     |                   |


## Performance Results
The following results are obtained on a machine with an `AMD Ryzen 7 5700g`
processor and `NVIDIA GeForce RTX 3060` GPU, by running the
`./runBenchmarks.sh` script in the root directory.

Note that some runtimes might not add up to the total time reported. This is
due to the intermediate times being rounded.

### Batched Histogram
|            | CPU Runtime | GPU Runtime | GPU Percentage | Speedup |
|:----------:|------------:|------------:|---------------:|--------:|
|  Transfer  |         N/A |       543ms |          97.8% |     N/A |
|    Fill    |      3383ms |        12ms |           2.2% |  281.9x |
|   Result   |         N/A |         0ms |           0.0% |     N/A |
|   Total    |      3383ms |       555ms |         100.0% |    6.1x |

### DiMuon
|               | CPU Runtime | GPU Runtime | GPU Percentage | Speedup |
|:-------------:|------------:|------------:|---------------:|--------:|
|   Transfer    |         N/A |       163ms |          58.3% |     N/A |
| Define + Fill |       788ms |       116ms |          41.7% |    6.8x |
|    Result     |         N/A |         0ms |           0.0% |     N/A |
|     Total     |       788ms |       278ms |         100.0% |    2.8x |

### FoldedWMass
|               | CPU Runtime | GPU Runtime | GPU Percentage | Speedup |
|:-------------:|------------:|------------:|---------------:|--------:|
|   Transfer    |         N/A |         0ms |           0.0% |     N/A |
| Define + Fill |     17792ms |       172ms |          98.9% |  103.4x |
|    Result     |         N/A |         1ms |           0.6% |     N/A |
|     Total     |     17792ms |       174ms |         100.0% |  102.3x |
