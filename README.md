# improc

`improc` is a small C++ image-processing library built around a templated `Image<T>` container, PNM file I/O, CPU image operations, and an optional CUDA path for GPU-resident `uint8_t` images.

## What is implemented

- CPU image container and utilities
- PGM/PPM read and write helpers
- Point operations: `invert`, `threshold`, `histogramEqualization`
- CPU filters: `convolve`, `medianFilter`, `gaussianBlur`, `sobelFilter`, `sharpeningFilter`, `laplacianFilter`
- CPU morphology: `erode`, `dilate`, `open`, `close`, `morphologicalGradient`, `topHat`, `bottomHat`
- CUDA image container: `cuda::GpuImage<uint8_t>`
- CUDA filters: `cuda::convolve`, `cuda::medianFilter`, `cuda::gaussianBlur`, `cuda::sobelFilter`, `cuda::sharpeningFilter`, `cuda::laplacianFilter`
- CUDA morphology: `cuda::erode`, `cuda::dilate`, `cuda::open`, `cuda::close`, `cuda::morphologicalGradient`, `cuda::topHat`, `cuda::bottomHat`

## CUDA notes

- Build CUDA support with `make BUILD_WITH_CUDA=1`
- The current CUDA implementations are explicitly instantiated for `uint8_t`
- CUDA morphology requires single-channel images
- The CUDA median filter currently supports kernels up to `15x15`
- CUDA support lives in [include/cuda/filters.cuh](/home/ben/improc/include/cuda/filters.cuh), [include/cuda/morph.cuh](/home/ben/improc/include/cuda/morph.cuh), [src/cuda/filters.cu](/home/ben/improc/src/cuda/filters.cu), and [src/cuda/morph.cu](/home/ben/improc/src/cuda/morph.cu)

## Build

```bash
make
make test
make clean
make static_analysis
make BUILD_WITH_CUDA=1
```

## Test status

The CPU test suite currently passes:

- `97` tests across `7` test suites via `make test` and `./bin/test`

The CUDA build path currently compiles successfully with:

- `make BUILD_WITH_CUDA=1`

There is not yet a dedicated CUDA test suite in `tests/`.

## Project layout

```text
include/
  image.hpp
  filters.hpp
  morph.hpp
  point_ops.hpp
  pnm_funcs.hpp
  pbm_funcs.hpp
  utils.hpp
  cuda/
    gpu_image.cuh
    gpu_utils.cuh
    filters.cuh
    morph.cuh
src/
  image.cpp
  filters.cpp
  morph.cpp
  point_ops.cpp
  pbm_funcs.cpp
  utils.cpp
  main.cpp
  cuda/
    gpu_image.cu
    gpu_utils.cu
    filters.cu
    morph.cu
tests/
images/
Makefile
```

## Minimal CPU example

```cpp
#include "filters.hpp"
#include "morph.hpp"
#include "pnm_funcs.hpp"

Image<uint8_t> input;
readPNM("images/sampleGRAY.pgm", input);

Image<uint8_t> blurred(input.rows(), input.cols(), input.channels());
gaussianBlur(input, blurred, 1.0f);

Image<uint8_t> opened(input.rows(), input.cols(), input.channels());
open(input, opened, 3);

savePNM("images/out.pgm", blurred);
```

## Minimal CUDA example

```cpp
#include "cuda/filters.cuh"
#include "cuda/gpu_image.cuh"
#include "cuda/morph.cuh"

Image<uint8_t> input;
readPNM("images/sampleGRAY.pgm", input);

cuda::GpuImage<uint8_t> d_input(input);
cuda::GpuImage<uint8_t> d_output(input.rows(), input.cols(), input.channels());

d_input.upload(input);
cuda::gaussianBlur(d_input, d_output);
d_output.download(input);
```

## Current gaps

- No CUDA unit tests yet
- Public PNM naming is slightly messy because the implementation file is still named `pbm_funcs.*`
- CUDA support is narrower than the CPU API and currently focused on `uint8_t`
- The demo program in [src/main.cpp](/home/ben/improc/src/main.cpp) still uses hard-coded sample-image paths
