# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **EPA SWMM (Storm Water Management Model) Solver** - a dynamic hydrology-hydraulic water quality simulation model for urban stormwater runoff. The solver is written in C and maintained by the US EPA Office of Research and Development.

- **Current Version:** 5.2.4
- **Language:** C (with some C++ for CMake configuration)
- **License:** Public Domain
- **Main Branch:** develop

## Build System

The project uses **CMake** (minimum version 3.13) as its build system.

### Basic Build Commands

**Standard Build (Linux/macOS):**
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

**Windows Build:**
```bash
mkdir build
cd build
cmake -G "Visual Studio 16 2019" .. -A x64
cmake --build . --config Release
```

**Build with Tests:**
```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build --config Debug
```

### Build Options

- `BUILD_TESTS=ON` - Builds component tests (requires Boost)
- `BUILD_DEF=ON` - Builds library with .def file interface for backward compatibility

### Build Outputs

The build produces:
- **swmm5** library (`swmm5.dll` on Windows, `libswmm5.so` on Linux/macOS) - The computational engine
- **runswmm** executable - Command line interface
- **swmm-output** library - Binary output file API
- All outputs are placed in `build/bin/<CONFIGURATION>/`

## Testing

### Unit Tests

Unit tests use the Boost Test framework and are located in `tests/`.

**Run Unit Tests:**
```bash
ctest --test-dir build -C Debug --output-on-failure
```

**Run Specific Test:**
```bash
build/bin/Debug/test_output
```

The CI/CD pipeline runs both unit tests and regression tests (via nrtest framework).

## Code Architecture

### Three-Layer Structure

The codebase is organized into three main components:

1. **`src/solver/`** - Core SWMM computational engine (~54 C source files)
   - Builds into the `swmm5` shared library
   - Contains all hydraulic/hydrologic simulation logic
   - Public API defined in `include/swmm5.h`
   - Main entry points: `swmm_run()`, `swmm_open()`, `swmm_start()`, `swmm_step()`, `swmm_end()`, `swmm_close()`

2. **`src/outfile/`** - Binary output file reader/writer library
   - Builds into the `swmm-output` shared library
   - Independent API for reading SWMM binary output files
   - Public API in `include/swmm_output.h`

3. **`src/run/`** - Command line executable
   - Builds into `runswmm` executable
   - Simple wrapper around the swmm5 library
   - Usage: `runswmm <input_file> <report_file> [output_file]`

### Key Solver Modules

The solver is modularized by functionality (each has corresponding .c/.h files):

- **Hydrology:** `subcatch.c`, `runoff.c`, `infil.c`, `gwater.c`, `snow.c`, `landuse.c`
- **Hydraulics:** `flowrout.c`, `dynwave.c`, `dwflow.c`, `kinwave.c`, `routing.c`
- **Components:** `node.c`, `link.c`, `conduit.c`, `pump.c`, `orifice.c`, `weir.c`, `outlet.c`
- **Input/Output:** `input.c`, `output.c`, `report.c`, `project.c`
- **Routing Objects:** `culvert.c`, `forcmain.c`, `street.c`, `inlet.c`
- **Control Systems:** `controls.c`, `rule.c`, `mathexpr.c`
- **Utilities:** `datetime.c`, `hash.c`, `mempool.c`, `massbal.c`, `stats.c`

### Code Organization Pattern

All solver modules follow a consistent header inclusion pattern (defined in `headers.h`):
1. `macros.h` - Common macros and constants
2. `objects.h` - Data structure definitions
3. `globals.h` - Global variable declarations
4. `funcs.h` - Function prototypes
5. `error.h` - Error codes
6. `text.h` - String constants
7. `keywords.h` - Input file keywords

**IMPORTANT:** The order of includes in `headers.h` must not be changed.

### SWMM Object Model

SWMM simulates stormwater systems using these primary object types:

- **SUBCATCH** (Subcatchments) - Surface areas that generate runoff
- **NODE** - Junction points (junctions, outfalls, storage units, dividers)
- **LINK** - Flow connectors (conduits, pumps, orifices, weirs, outlets)
- **GAGE** - Rainfall gauges
- **POLLUTANT** - Water quality constituents

All objects are indexed arrays accessed by integer indices. Use `swmm_getIndex()` to convert names to indices.

## API Usage

The SWMM5 API supports two usage patterns:

**Pattern 1: Complete Simulation**
```c
swmm_run(inputFile, reportFile, binaryFile);
```

**Pattern 2: Step-by-Step Control**
```c
swmm_open(inputFile, reportFile, binaryFile);
swmm_start(saveFlag);
do {
    swmm_step(&elapsedTime);
    // Custom logic here
} while (elapsedTime < simulationDuration);
swmm_end();
swmm_report();
swmm_close();
```

## Dependencies

- **OpenMP** - Required for parallel processing (solver only)
- **Boost Test** - Optional, required only for unit tests (version 1.67.0+)
- **CMake** - Version 3.13 or higher
- **C Compiler** - Visual Studio 2017+ (Windows) or GCC/Clang (Linux/macOS)

## Platform-Specific Notes

### Windows
- Uses `__declspec(dllexport)` for library exports
- Default build generates 64-bit binaries
- MSVC-specific optimizations enabled in Release builds (`/GL`, `/fp:fast`, `/LTCG`)

### Linux/macOS
- Position-independent code (`-fPIC`) enabled by default
- Math library (`-lm`) linked automatically
- On macOS, OpenMP requires special handling (see `src/solver/CMakeLists.txt`)

## Running SWMM

**Command Line:**
```bash
runswmm input.inp report.rpt output.out
```

**Help:**
```bash
runswmm --help
```

**Version:**
```bash
runswmm --version
```

## Installation

CMake installs to `build/install/` by default:
```bash
cmake --install build
```

Installed components:
- `bin/` - Executables and shared libraries
- `include/` - Public header files (`swmm5.h`, `swmm_output.h`)
- `lib/` - Static libraries (if built)
- `cmake/` - CMake config files for downstream projects

## GPU Acceleration (SWMM-GPU)

This repository includes ongoing work to add GPU acceleration support via CUDA.

**Project Name:** SWMM-GPU
**Target Hardware:** NVIDIA GPUs with CUDA support (Compute Capability 6.0+)
**Build Option:** `cmake -DBUILD_CUDA=ON`

**Status:** Under development - see task list for GPU parallelization of:
- Dynamic wave flow routing (`dynwave.c` - link flows and node depths)
- Picard iteration convergence checking

**Expected Performance:** 5-15x speedup for large models (1000+ links) on modern GPUs.

## External Resources

- Official SWMM Website: http://www.epa.gov/water-research/storm-water-management-model-swmm
- Regression Test Suite: https://github.com/USEPA/swmm-nrtestsuite
- CI Tools: https://github.com/USEPA/swmm-ci-tools
