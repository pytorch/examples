#include <torch/script.h>

#include <sstream>

// ElementwiseInterpreter is a class that takes in a list of Instructions
// (represented as triples (op, inputs, outputs)) and executes them in a
// C++-based interpreter loop. For brevity, this interpreter only supports
// two operations: element-wise add and element-wise mul.
struct ElementwiseInterpreter : torch::CustomClassHolder {
  using InstructionType =
      std::tuple<std::string /*op*/, std::vector<std::string> /*inputs*/,
                 std::string /*output*/>;

  ElementwiseInterpreter() {}

  // Load a list of instructions into the interpreter. As specified above,
  // instructions specify the operation (currently support "add" and "mul"),
  // the names of the input values, and the name of the single output value
  // from this instruction
  void setInstructions(std::vector<InstructionType> instructions) {
    instructions_ = std::move(instructions);
  }

  // Add a constant. The interpreter maintains a set of constants across
  // calls. They are keyed by name, and constants can be referenced in
  // Instructions by the name specified
  void addConstant(const std::string &name, at::Tensor value) {
    constants_.insert_or_assign(name, std::move(value));
  }

  // Set the string names for the positional inputs to the function this
  // interpreter represents. When invoked, the interpreter will assign
  // the positional inputs to the names in the corresponding position in
  // input_names.
  void setInputNames(std::vector<std::string> input_names) {
    input_names_ = std::move(input_names);
  }

  // Specify the output name for the function this interpreter represents. This
  // should match the "output" field of one of the instructions in the
  // instruction list, typically the last instruction.
  void setOutputName(std::string output_name) {
    output_name_ = std::move(output_name);
  }

  // Invoke this interpreter. This takes a list of positional inputs and returns
  // a single output. Currently, inputs and outputs must all be Tensors.
  at::Tensor __call__(std::vector<at::Tensor> inputs) {
    // Environment to hold local variables
    std::unordered_map<std::string, at::Tensor> environment;

    // Load inputs according to the specified names
    if (inputs.size() != input_names_.size()) {
      std::stringstream err;
      err << "Expected " << input_names_.size() << " inputs, but got "
          << inputs.size() << "!";
      throw std::runtime_error(err.str());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      environment[input_names_[i]] = inputs[i];
    }

    if (!output_name_) {
      throw std::runtime_error("Output name not specified!");
    }

    for (InstructionType &instr : instructions_) {
      // Retrieve all input values for this op
      std::vector<at::Tensor> inputs;
      for (const auto &input_name : std::get<1>(instr)) {
        // Operator output values shadow constants.
        // Imagine all constants are defined in statements at the beginning
        // of a function (a la K&R C). Any definition of an output value must
        // necessarily come after constant definition in textual order. Thus,
        // We look up values in the environment first then the constant table
        // second to implement this shadowing behavior
        if (environment.find(input_name) != environment.end()) {
          inputs.push_back(environment.at(input_name));
        } else if (constants_.find(input_name) != constants_.end()) {
          inputs.push_back(constants_.at(input_name));
        } else {
          std::stringstream err;
          err << "Instruction referenced unknown value " << input_name << "!";
          throw std::runtime_error(err.str());
        }
      }

      // Run the specified operation
      at::Tensor result;
      const auto &op = std::get<0>(instr);
      if (op == "add") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for add op!");
        }
        result = inputs[0] + inputs[1];
      } else if (op == "mul") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for mul op!");
        }
        result = inputs[0] * inputs[1];
      } else {
        std::stringstream err;
        err << "Unknown operator " << op << "!";
        throw std::runtime_error(err.str());
      }

      // Write back result into environment
      const auto &output_name = std::get<2>(instr);
      environment[output_name] = std::move(result);
    }

    if (!environment.count(*output_name_)) {
      std::stringstream err;
      err << "Execution expected an output value with name ";
      err << *output_name_;
      err << " but no instruction produced a value with that name!";
      throw std::runtime_error(err.str());
    }

    return environment.at(*output_name_);
  }

  // Ser/De infrastructure. See
  // https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html#defining-serialization-deserialization-methods-for-custom-c-classes
  // for more info.

  // This is the type we will use to marshall information on disk during
  // ser/de. It is a simple tuple composed of primitive types and simple
  // collection types like vector, optional, and dict.
  using SerializationType =
      std::tuple<std::vector<std::string> /*input_names_*/,
                 c10::optional<std::string> /*output_name_*/,
                 c10::Dict<std::string, at::Tensor> /*constants_*/,
                 std::vector<InstructionType> /*instructions_*/
                 >;

  // This function yields the SerializationType instance for `this`.
  SerializationType __getstate__() const {
    return SerializationType{input_names_, output_name_, constants_,
                             instructions_};
  }

  // This function will create an instance of `ElementwiseInterpreter` given
  // an instance of `SerializationType`.
  static c10::intrusive_ptr<ElementwiseInterpreter>
  __setstate__(SerializationType state) {
    auto instance = c10::make_intrusive<ElementwiseInterpreter>();
    std::tie(instance->input_names_, instance->output_name_,
             instance->constants_, instance->instructions_) = std::move(state);
    return instance;
  }

  // Class members
  std::vector<std::string> input_names_;
  c10::optional<std::string> output_name_;
  c10::Dict<std::string, at::Tensor> constants_;
  std::vector<InstructionType> instructions_;
};

// Register ElementwiseInterpreter as callable from Python
// and TorchScript.
TORCH_LIBRARY(NativeInterpretation, m) {
  m.class_<ElementwiseInterpreter>("ElementwiseInterpreter")
      .def(torch::init<>())
      .def("set_instructions", &ElementwiseInterpreter::setInstructions)
      .def("add_constant", &ElementwiseInterpreter::addConstant)
      .def("set_input_names", &ElementwiseInterpreter::setInputNames)
      .def("set_output_name", &ElementwiseInterpreter::setOutputName)
      .def("__call__", &ElementwiseInterpreter::__call__)
      .def_pickle(
          /* __getstate__ */
          [](const c10::intrusive_ptr<ElementwiseInterpreter> &self) {
            return self->__getstate__();
          },
          /* __setstate__ */
          [](ElementwiseInterpreter::SerializationType state) {
            return ElementwiseInterpreter::__setstate__(std::move(state));
          });
}
