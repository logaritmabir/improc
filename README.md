# improc - Image Processing Library

A modern C++ image processing library with support for reading, writing, and filtering images in **PNM format only** (PGM for grayscale, PPM for color).

## Overview

`improc` is a lightweight, template-based image processing library that provides:

- **Image Class**: A generic, template-based container for image data supporting multiple channels and flexible data types
- **PNM I/O**: Functions to read and write images exclusively in Portable PixMap (PPM) and Portable GrayMap (PGM) formats
- **Image Filtering**: Convolution-based image filters for applying transformations like blurring, edge detection, and more
- **Color Channel Operations**: Extract individual RGB channels or convert to grayscale
- **Type Flexibility**: Template-based design allows working with different data types (uint8_t, float, etc.)

## Project Structure

```
improc/
├── include/              # Header files
│   ├── image.hpp        # Image container class
│   ├── filters.hpp      # Image filtering functions
│   └── pnm_funcs.hpp   # PNM file I/O functions
├── src/                 # Implementation files
│   ├── image.cpp
│   ├── filters.cpp
│   ├── pnm_funcs.cpp
│   └── main.cpp         # Example usage
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
- `convertP6toP5()` - Convert between PPM color formats

### Filters (`filters.hpp`)
Image processing operations:
- `convolve()` - Apply 2D convolution with custom kernels

## Building

```bash
make              # Build the main executable
make clean        # Remove build artifacts
```

## Usage

The main program demonstrates loading a grayscale image, applying a convolution kernel, and saving the result:

```cpp
Image<uint8_t> sample_img;
readPNM("../images/sampleGRAY.pgm", sample_img);

Image<uint8_t> output_img = sample_img;
std::vector<std::vector<uint8_t>> kernel = {{1,0,1}, {1,0,1}, {1,0,1}};
uint8_t divideBy = 9;
convolve(sample_img, output_img, kernel, divideBy);

savePNM("../images/output.pgm", output_img);
```

## Requirements

- C++11 or later
- GCC/Clang compiler
- GoogleTest (for running unit tests)

## Example Images

The `images/` directory contains sample test images:
- `sampleGRAY.pgm` - Sample grayscale image
- `sampleRGB.ppm` - Sample color image
- `output.pgm` - Output from the convolution example