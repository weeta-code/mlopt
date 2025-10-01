#pragma once

/**
*     IR & GraphModule for MLOpt ver 0
*
*     @file graph.h
*     @brief Minimal IR + GraphModule public surface
*
*     invariants:
*     - ValueId and NodeId are stable within a GraphModule lifetime and not reused.
*     - TensorType uses -1 for unknown Dimensions.
*     - Mutations that affect topology must leave the graph verifiable via verify().
*     
*     Ownership model:
*       - GraphModule owns all nodes/values/attrs.
*       - All implementation details live in .cpp (attrs, storage, maps, use-lists, etc.).
*       - Consumer interact via IDs and read-only snapshot views (ValueView/NodeView)
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
  
  // Convenience sake only
  bool is_scalar() const { return shape.empty(); }
  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
};

// Stable identifiers (handles)

using ValueId = uint32_t;
using NodeId = uint32_t;
inline constexpr ValueId  kInvalidValueId  = std::numeric_limits<ValueId>::max();
inline constexpr NodeId   kInvalidNodeId    = std::numeric_limits<NodeId>::max();

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

// Status-lite
// Keep public surface exception-light.

class Status {
public:
  static Status OK() { return Status(); }
  static Status Error(std::string msg) { return Status(std::move(msg)); }

  bool ok() const { return msg_.empty(); }
  const std::string& message() const { return msg_; }

private:
  explicit Status(std::string msg = {}) : msg_(std::move(msg)) {}
  std::string msg_;
};

// Core IR container

class GraphModule {
public:
  GraphModule();
  ~GraphModule();

  GraphModule(GraphModule&&) noexcept = default;
  GraphModule& operator=(GraphModule&&) noexcept = default;

  // Non-copyable (internal IDS/indices to make copying error-prone for now)
  GraphModule(const GraphModule&) = delete;
  GraphModule& operator=(const GraphModule&) = delete;

  // Constructor
  ValueId add_input(const std::string& name, const TensorType& type);

  NodeId add_node(const std::string& op,
                  const std::vector<ValueId>& inputs,
                  const AttrMap& attrs = {});

  ValueId add_output(ValueId from_value);

  // helper for tests, returns ValueId of produced constant.
  ValueId add_const_scalar(DType dtype, const AttrValue& scalar);

  // Query
  
  // Deterministic iteration order
  std::vector<NodeId> nodes() const;
  std::vector<ValueId> inputs() const;
  std::vector<ValueId> outputs() const;

  NodeView get_node(NodeId id) const;
  ValueView get_value(ValueId id) const;

  size_t num_nodes() const;
  size_t num_values() const;

  // Transforms

  [[nodiscard]] Status replace_node(NodeId node, const std::string& new_op,
                      const std::vector<ValueId>& new_inputs,
                      const AttrMap& new_attrs,
                      NodeId* out_new_node = nullptr);

  [[nodiscard]] Status remove_node(NodeId node);

  size_t replace_all_uses(ValueId from_value, ValueId to_value);

  [[nodiscard]] Status topological_sort(std::vector<NodeId>* out) const;

  // Integrity

  [[nodiscard]] Status verify() const;

  // I/O

  [[nodiscard]] Status to_json(const std::string& path) const;

  [[nodiscard]] static Status from_json(const std::string& path, GraphModule* out);

  // Diagnostic
  const std::string& last_error() const;

private:
  // PIMPL to be implemented
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Pass Infra (surface)

class Pass {
public: 
  virtual ~Pass() = default;
  virtual const char* name() const = 0;
  virtual bool run(GraphModule& m) = 0;
};

class PassManager {
public:
  PassManager();
  ~PassManager();

  PassManager(PassManager&&) noexcept = default;
  PassManager& operator=(PassManager&&) noexcept = default;

  PassManager(const PassManager&) = delete;
  PassManager& operator=(const PassManager&) = delete;

  void add(Pass* pass);
  void clear();

  void run(GraphModule& m);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};


} // namespace mlopt::ir
