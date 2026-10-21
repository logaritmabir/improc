# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Planning Complete

### Planned for v0.1.0 (Stabilization Release)
- **Build System**: Add static library target (`libimproc.a`), `install`/`uninstall`/`dist` targets, `pkg-config` support (`.pc` file)
- **Versioning**: Embed version from `VERSION` file into binary (`--version` flag)
- **CUDA Tests**: Numerical equivalence tests (CPU vs GPU) for all filters and morphology operations
- **API Cleanup**: Rename `pbm_funcs.hpp` → `pnm_funcs.hpp`, move public headers to `include/improc/`, add `namespace improc`
- **Documentation**: Update README with install/test instructions, fix Doxygen `@ref` links, fix hardcoded paths in demo
- **Code Quality**: Enable static analysis in CI, add compiler warnings

### Known Limitations (carried forward)
- CUDA implementations only instantiated for `uint8_t`
- CUDA morphology requires single-channel images
- CUDA median filter limited to 15x15 kernel
- PNM format only (no PNG/JPEG/TIFF)
- Demo program uses hard-coded sample image paths

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

[Unreleased]: https://github.com/ben/improc/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/ben/improc/releases/tag/v0.0.1