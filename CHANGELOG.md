# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-10-11

### Added
- Initial project structure with CPU and CUDA image processing operations
- `Image<T>` template container with multi-channel support
- PNM I/O (PGM/PPM read/write) via `pnm_funcs.hpp`
- Point operations: `invert`, `threshold`, `histogramEqualization`
- CPU filters: `convolve`, `medianFilter`, `gaussianBlur`, `sobelFilter`, `sharpeningFilter`, `laplacianFilter`
- CPU morphology: `erode`, `dilate`, `open`, `close`, `morphologicalGradient`, `topHat`, `bottomHat`
- CUDA support (`BUILD_WITH_CUDA=1`):
  - `cuda::GpuImage<uint8_t>` container with upload/download
  - CUDA filters: `convolve`, `medianFilter`, `gaussianBlur`, `sobelFilter`, `sharpeningFilter`, `laplacianFilter`
  - CUDA morphology: `erode`, `dilate`, `open`, `close`, `morphologicalGradient`, `topHat`, `bottomHat`
  - NPP convolution wrapper (`npp::convolution`)
- GoogleTest test suite (97 CPU tests passing)
- Doxygen documentation configuration
- MIT License

### Known Limitations
- CUDA implementations only instantiated for `uint8_t`
- CUDA morphology requires single-channel images
- CUDA median filter limited to 15x15 kernel
- No CUDA unit tests yet (planned for v0.1.0)
- PNM format only (no PNG/JPEG/TIFF)
- Demo program uses hard-coded sample image paths

[0.0.1]: https://github.com/ben/improc/releases/tag/v0.0.1