#pragma once

/**
*     IR & GraphModule for MLOpt ver 0
*
*     invariants:
*     
*     Ownership model:
*       - GraphModule owns all nodes/values/attrs.
*       - All implementation details live in .cpp (attrs, storage, maps, use-lists, etc.).
*/ 

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

namespace mlopt::ir {

// Versioning

constexpr int MLOPT_IR_VERSION = 1;

// Scalar / Type system

enum class DType : uint8_t {
  I8, I16, I32, I64,
  U8, U16, U32, U64,
  F16, BF16, F32, F64,
  BOOL
};
}
