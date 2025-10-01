# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

MLOpt is an ML compiler and inference optimization framework in early development. The end goal is to build an **inference optimization pipeline** that leverages **TensorRT** and **ONNX** for accelerated ML inference.

The graph IR serves as an intermediate representation that allows ONNX models to be loaded, analyzed, and optimized through custom transformations before being compiled for inference. This provides a flexible layer between ONNX model format and TensorRT optimization, enabling fine-grained control over graph-level optimizations.

The project is currently implementing a core graph abstraction module that serves as the foundation for uploading and processing ML models using a custom "graph" dialect that interfaces with ONNX.

## Architecture

### Core Components

**Graph IR System** (`include/graph.h`, `src/graph.cpp`)
- `GraphModule`: Central container for the IR graph representation
- `ValueView`/`NodeView`: Read-only snapshot views for graph inspection
- `TensorType`: Type system supporting various data types (I8-I64, U8-U64, F16, BF16, F32, F64, BOOL)
- Stable identifier system using `ValueId` and `NodeId` handles
- PIMPL design pattern with implementation details hidden in `.cpp`

**Pass Infrastructure**
- `Pass`: Base class for graph transformations and optimizations  
- `PassManager`: Orchestrates multiple passes over graph representations
- Support for topological sorting and graph verification

**Key Design Principles**
- Stable IDs: ValueId and NodeId remain stable throughout GraphModule lifetime
- Unknown dimensions represented as -1 in TensorType
- Exception-light design with Status class for error handling
- Ownership model: GraphModule owns all nodes/values/attributes
- Consumer interaction via IDs and read-only views only

### Directory Structure
```
mlopt/
├── include/
│   └── graph.h          # Public API and IR definitions
├── src/  
│   └── graph.cpp        # Implementation (currently empty - under development)
└── README.md           # Project overview
```

## Development Status

The project is in active development ("Dev stage") with the graph abstraction module being the current focus. The header file defines the complete public API, but implementation in `src/graph.cpp` is still pending.

## Development Commands

**Note**: This project does not currently have a build system configured. You will need to set up compilation infrastructure.

**Typical C++ Development Setup Needed:**
```bash
# Create a basic Makefile or CMakeLists.txt for compilation
# Example compilation command (once build system is added):
g++ -std=c++17 -I include src/*.cpp -o mlopt

# For development with tests:
# Add test framework (e.g., Google Test, Catch2)
# Create tests/ directory structure
```

**Git Workflow:**
```bash
git log --oneline                    # View commit history
git status                          # Check current changes
git add include/ src/               # Stage changes
git commit -m "feat: description"   # Commit with conventional format
```

## Development Notes

- The codebase follows a conventional commit style (`feat:`, `chore:`, etc.)
- C++17 features are expected based on the code patterns
- The project draws inspiration from MLIR architecture
- Implementation follows PIMPL idiom for clean public interfaces
- **Target Pipeline**: ONNX → Graph IR → Optimizations → TensorRT compilation
- The Graph IR is specifically designed as an interface layer for ONNX models
- Focus on enabling custom node-level optimizations before TensorRT compilation

## Next Development Steps

Based on the current state and the inference optimization pipeline goal:
1. Implementing the GraphModule methods in `src/graph.cpp`
2. Adding build system (CMakeLists.txt or Makefile) with ONNX dependencies
3. Creating test infrastructure and test cases for graph operations
4. Developing the pass system for graph optimizations (the core value proposition)
5. Adding ONNX model loading/parsing into Graph IR format
6. Implementing ONNX-to-Graph IR conversion utilities
7. Building optimization passes for common inference patterns
8. Adding TensorRT backend integration for optimized inference
