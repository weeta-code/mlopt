#include "../include/graph.hpp"
#include <utility>

namespace mlopt::ir {

struct GraphModule::Impl {
  std::string last_err;

  // TODO : allocate storage for nodes/values/use-lists, deterministic id vectors and everything else
};

GraphModule::GraphModule() : impl_(std::make_unique<Impl>()) {}
GraphModule::~GraphModule() = default;

ValueId GraphModule::add_input(const std::string& name, const TensorType& type) {
  // TODO : should mark a Value as a graph input and return that ValueId, allocated ValueId & store the name/type
  return ValueId{0}; // placehold
}

NodeId GraphModule::add_node(const std::string& op, const std::vector<ValueId>& inputs, const AttrMap& attrs) {
  // TODO : adds a compute node, return that new node, allocate it as well & store
  // necessary info (op, inputs, attrs)
  return NodeId{0};
}

ValueId GraphModule::add_output(ValueId v) { return v; } // revisit, should mark existing value as graph output?

ValueId GraphModule::add_const_scalar(DType, const AttrValue&) {
  // TODO : 

  return ValueId{0};
}

std::vector<NodeId> GraphModule::nodes() const { return {}; }
std::vector<NodeId> GraphModule::inputs() const { return {}; }
std::vector<NodeId> GraphModule::outputs() const { return {}; }

NodeView GraphModule::get_node(NodeId) const { return {}; }
ValueView GraphModule::get_value(ValueId) const { return {}; }

size_t GraphModule::num_nodes() const { return 0; }
size_t GraphModule::num_values() const { return 0; }

Status GraphModule::replace_node(NodeId node, const std::string& new_op, 
                                 const std::vector<ValueId>& new_inputs,
                                 const AttrMap& new_attrs, NodeId*) {
  return Status::OK();
}

Status GraphModule::topological_sort(std::vector<NodeId>* out) const;
}
