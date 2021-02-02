# Converting PyTorch Code to a Native Runtime With FX and TorchScript Custom Classes

In this example, we are going to build a pipeline that does the following things:

1. Converts (or “lowers”) code in a PyTorch module into another representation (we will define the representation within the example)
2. Registers an interpreter for that code representation that can be used in TorchScript or Python
3. Wrap the converted code into a format that can still be used in TorchScript compilation.

We are going to build up a trivial interpreter for this example, but you can imagine extending the same process to work with more sophisticated backends, ones which may do code optimization or offloading to an accelerator.

We will be using [TorchScript custom classes](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) to expose this Interpreter to Python and TorchScript. You may want to review that tutorial and documentation before reading this example project.

### Defining the Interpreter

We define the interpreter in `interpreter.cpp`. This interpreter is very limited: it only supports two element-wise operations (`add` and `mul`) and it only supports `Tensor` values. When this interpreter runs code, it iterates through the list of instructions and simply calls the appropriate PyTorch operator from C++.

To build the interpreter into a shared-object file to be loaded in for use, use the following commands from this example’s root:


```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
$ make -j
```

After the build finishes, you should see `build/libinterpreter.so` (or with a different extension depending on your OS). We will use this dynamic library next when we load it up into a process to be used in execution.

### Defining the Transformation

We define the code that transforms a `PyTorch` module to the format the interpreter understands in `use_interpreter.py`. Note that that file loads in the shared object we built in the previous step via a `torch.classes.load_library` call. `use_interpreter.py` contains driver code and the end that can be directly run to test the lowering transformation.

### Questions, Comments, Feedback

Please direct questions and discussion to the [PyTorch forums](https://discuss.pytorch.org/). To report any issues with PyTorch (including FX and custom classes), please use the [issue tracker](https://github.com/pytorch/pytorch/issues).
