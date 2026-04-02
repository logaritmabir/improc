# improc - Image Processing Library

A modern C++ image processing library with support for reading, writing, and filtering images in **PNM format only** (PGM for grayscale, PPM for color), with optional GPU acceleration via CUDA.

## Overview

`improc` is a lightweight, template-based image processing library that provides:

- **Image Class**: A generic, template-based container for image data supporting multiple channels and flexible data types
- **PNM I/O**: Functions to read and write images exclusively in Portable PixMap (PPM) and Portable GrayMap (PGM) formats
- **Point Operations**: Per-pixel transformations like inversion and thresholding
- **Image Filtering**: Convolution-based filters including blur, median filter, edge detection, sharpening, and Laplacian
- **Morphological Operations**: Erosion and dilation for shape analysis
- **Color Operations**: Extract RGB channels and convert to grayscale
- **GPU Acceleration**: CUDA-based convolution for high-performance processing
- **Type Flexibility**: Template-based design allows working with different data types (uint8_t, float, etc.)

## Project Structure

```
improc/
├── include/              # Header files
│   ├── image.hpp        # Image container class
│   ├── filters.hpp      # Image filtering functions
│   ├── point_ops.hpp    # Point-wise operations
│   ├── morph.hpp        # Morphological operations
│   ├── utils.hpp        # Utility functions
│   ├── pnm_funcs.hpp   # PNM file I/O functions
│   └── cuda/            # CUDA support
│       ├── gpu_image.cuh # GPU image container
│       └── kernels.cuh   # CUDA convolution kernels
├── src/                 # Implementation files
│   ├── image.cpp
│   ├── filters.cpp
│   ├── point_ops.cpp
│   ├── morph.cpp        # Morphological operations
│   ├── utils.cpp
│   ├── pnm_funcs.cpp
│   ├── main.cpp         # Example usage
│   └── cuda/            # CUDA implementations
│       ├── gpu_image.cu
│       └── kernels.cu
├── tests/               # Unit tests
│   └── image_test.cpp
├── images/              # Sample images and outputs
├── bin/                 # Compiled executables
└── Makefile             # Build configuration
```

## Key Components

### Image Class (`image.hpp`)
A template-based container for storing image data:
- Support for multiple channels (grayscale, RGB, etc.)
- Random access to pixels via `at()` or `operator()`
- Move and copy semantics
- Iterator support for STL compatibility

### PNM Functions (`pnm_funcs.hpp`)
I/O operations for image files:
- `readPNM()` - Load images from PGM (grayscale) or PPM (color) files
- `savePNM()` - Write images to PGM or PPM format
- `extractChannel()` - Extract individual R, G, or B channels
- `rgbToGrayscale()` - Convert RGB to grayscale using luminosity formula

### Point Operations (`point_ops.hpp`)
Per-pixel transformations:
- `invert()` - Invert pixel values (photographic negative)
- `threshold()` - Apply binary thresholding

### Filters (`filters.hpp`)
Image processing operations:
- `convolve()` - Apply 2D convolution with custom kernels
- `medianFilter()` - Apply median filter for noise reduction
- `gaussianBlur()` - Apply Gaussian blur for smoothing
- `sobelFilter()` - Apply Sobel edge detection
- `sharpeningFilter()` - Enhance image details and edges
- `laplacianFilter()` - Apply Laplacian edge detection

### Morphological Operations (`morph.hpp`)
Shape-based image processing:
- `erode()` - Perform morphological erosion
- `dilate()` - Perform morphological dilation

### CUDA Support (`cuda/`)
GPU-accelerated processing:
- `GpuImage` - GPU-side image container for memory management
- `cuda::convolve()` - High-performance GPU convolution

## Building

```bash
make              # Build the main executable
make clean        # Remove build artifacts
make run          # Build and run the example
```

## Usage

### Basic Example

```cpp
#include "pnm_funcs.hpp"
#include "filters.hpp"

Image<uint8_t> img;
readPNM("input.pgm", img);

Image<uint8_t> blurred;
gaussianBlur(img, blurred, 2.0f);

Image<uint8_t> edges;
sobelFilter(img, edges);

Image<uint8_t> inverted;
invert(img, inverted);

Image<uint8_t> binary;
threshold(img, binary, 128);

savePNM("output.pgm", blurred);
```

### CUDA Example

```cpp
#include "cuda/gpu_image.cuh"
#include "cuda/kernels.cuh"

cuda::GpuImage<uint8_t> d_input(img);
cuda::GpuImage<uint8_t> d_output(img);

d_input.upload(img);

const float kernel[] = {
    0.0625f, 0.1250f, 0.0625f,
    0.1250f, 0.2500f, 0.1250f,
    0.0625f, 0.1250f, 0.0625f
};

cuda::convolve(d_input, d_output, kernel, 3);
d_output.download(result);
```

## Requirements

- C++11 or later
- GCC/Clang compiler
- CUDA Toolkit (for GPU acceleration)
- GoogleTest (for running unit tests)

## Example Images

The `images/` directory contains sample test images:
- `sampleGRAY.pgm` - Sample grayscale image
- `sampleRGB.ppm` - Sample color image
