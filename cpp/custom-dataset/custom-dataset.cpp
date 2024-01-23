#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>

struct Options {
  int image_size = 224;
  size_t train_batch_size = 8;
  size_t test_batch_size = 200;
  size_t iterations = 10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string datasetPath = "./dataset/";
  std::string infoFilePath = "info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, long>>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  const Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  Example get(size_t index) {
    std::string path = options.datasetPath + data[index].first;
    auto mat = cv::imread(path);
    assert(!mat.empty());

    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    auto R = torch::from_blob(
        channels[2].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);

    auto tdata = torch::cat({R, G, B})
                     .view({3, options.image_size, options.image_size})
                     .to(torch::kFloat);
    auto tlabel = torch::tensor(data[index].second, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};

std::pair<Data, Data> readInfo() {
  std::random_device randomDevice;
  std::mt19937 mersenneTwisterGenerator(randomDevice());
  Data train, test;

  std::ifstream stream(options.infoFilePath);
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;

    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else if (type == "test")
      test.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  std::shuffle(train.begin(), train.end(), mersenneTwisterGenerator);
  std::shuffle(test.begin(), test.end(), mersenneTwisterGenerator);
  return std::make_pair(train, test);
}

struct NetworkImpl : torch::nn::SequentialImpl {
  NetworkImpl() {
    using namespace torch::nn;

    auto stride = torch::ExpandingArray<2>({2, 2});
    torch::ExpandingArray<2> shape({-1, 256 * 6 * 6});
    push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Functional(torch::reshape, shape));
    push_back(Dropout());
    push_back(Linear(256 * 6 * 6, 4096));
    push_back(Functional(torch::relu));
    push_back(Dropout());
    push_back(Linear(4096, 4096));
    push_back(Functional(torch::relu));
    push_back(Linear(4096, 102));
    push_back(Functional(
        [](torch::Tensor input) { return torch::log_softmax(input, 1); }));
  }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(
    Network& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
  size_t index = 0;
  network->train();
  float Loss = 0, Acc = 0;

  for (auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();

    if (index++ % options.log_interval == 0) {
      auto end = std::min(data_size, (index + 1) * options.train_batch_size);

      std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                << std::endl;
    }
  }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size) {
  size_t index = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;

  for (const auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % options.log_interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}

int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  const auto data = readInfo();

  auto train_set =
      CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_set), options.train_batch_size);

  auto test_set =
      CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_set), options.test_batch_size);

  Network network;
  network->to(options.device);

  torch::optim::SGD optimizer(
      network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  for (size_t i = 0; i < options.iterations; ++i) {
    train(network, *train_loader, optimizer, i + 1, train_size);
    std::cout << std::endl;
    test(network, *test_loader, test_size);
    std::cout << std::endl;
  }

  return 0;
}
