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

struct TensorType {
  DType dtype{};
  std::vector<int64_t> shape; 

  bool is_scalar() const { return shape.empty(); }
  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
};

// Stable identifiers (handles)

using ValueId = uint32_t;
using NodeId = uint32_t;
inline constexpr ValueId  kInvalidValueId  = static_cast<ValueId>(-1);
inline constexpr NodeId   kInvalidNodeId    = static_cast<NodeId>(-1);

// Attributes (public surface)

struct AttrValue;
using AttrList  = std::vector<AttrValue>;
using AttrMap   = std::unordered_map<std::string, AttrValue>;

struct AttrValue {
  using Variant = std::variant<
    int64_t,
    double,
    std::string,
    bool,
    AttrList
  >;

  Variant value;

  AttrValue() = default;
  template <typename T>
  AttrValue(T v) : value(std::move(v)) {}
};

// Read Only

struct ValueView {
  ValueId id{kInvalidValueId};
  TensorType type{};
  NodeId produce{kInvalidNodeId}; // -1 means graph input
  std::string name;               // optional debug/original name
  bool is_input{false};
  bool is_output{false};
};

struct NodeView {
  NodeId id{kInvalidNodeId};
  std::string op;                 // for future onnx dialect ops
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  AttrMap attrs;                  // snapshot
};
};
