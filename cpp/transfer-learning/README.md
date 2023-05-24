# Transfer Learning on Dogs vs Cats Dataset using Libtorch and OpenCV

Transfer Learning on Dogs vs Cats dataset using PyTorch C++ API.

## Usage

For **training**:

1. Remove final layer of `ResNet18` pre-trained model and convert to `torch.jit` module: `python3 convert.py`.
2. Create build directory: `mkdir build && cd build`
3. `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`
4. `make`
5. Run training code: `./example <path_to_scripting_model>`

For **prediction**:

1. `cd build`
2. `./classify <path_image> <path_to_resnet18_model_without_fc_layer> <model_linear_trained>` : `./classify <path_image> ../resnet18_without_last_layer.pt model_linear.pt`

Detailed blog on applying Transfer Learning using Libtorch: https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/.
