#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

struct Net : torch::nn::Module {
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

struct Options {
  std::string data_root{"data"};
  int32_t batch_size{64};
  int32_t epochs{10};
  double lr{0.01};
  double momentum{0.5};
  bool no_cuda{false};
  int32_t seed{1};
  int32_t test_batch_size{1000};
  int32_t log_interval{10};
};

template <typename DataLoader>
void train(
    int32_t epoch,
    const Options& options,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % options.log_interval == 0) {
      std::cout << "Train Epoch: " << epoch << " ["
                << batch_idx * batch.data.size(0) << "/" << dataset_size
                << "]\tLoss: " << loss.template item<float>() << std::endl;
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::cout << "Test set: Average loss: " << test_loss
            << ", Accuracy: " << static_cast<double>(correct) / dataset_size
            << std::endl;
}

struct Normalize : public torch::data::transforms::TensorTransform<> {
  Normalize(float mean, float stddev)
      : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
  torch::Tensor operator()(torch::Tensor input) {
    return input.sub(mean_).div(stddev_);
  }
  torch::Tensor mean_, stddev_;
};

auto main() -> int {
  Options options;

  torch::manual_seed(options.seed);

  torch::DeviceType device_type;
  if (torch::cuda::is_available() && !options.no_cuda) {
    std::cout << "CUDA available! Training on GPU" << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(
          options.data_root, torch::data::datasets::MNIST::Mode::kTrain)
          .map(Normalize(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), options.batch_size);

  auto test_dataset =
      torch::data::datasets::MNIST(
          options.data_root, torch::data::datasets::MNIST::Mode::kTest)
          .map(Normalize(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(
      std::move(test_dataset), options.batch_size);

  torch::optim::SGD optimizer(
      model.parameters(),
      torch::optim::SGDOptions(options.lr).momentum(options.momentum));

  for (size_t epoch = 1; epoch <= options.epochs; ++epoch) {
    train(
        epoch,
        options,
        model,
        device,
        *train_loader,
        optimizer,
        train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
}
