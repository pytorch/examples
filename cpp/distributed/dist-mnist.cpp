#include <c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>
#include <iostream>

// Define a Convolutional Module
struct Model : torch::nn::Module {
  Model()
      : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
        conv2(torch::nn::Conv2dOptions(10, 20, 5)),
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
    x = torch::dropout(x, 0.5, is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

void waitWork(
    std::shared_ptr<c10d::ProcessGroupMPI> pg,
    std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works) {
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
  }
}

int main(int argc, char* argv[]) {
  // Creating MPI Process Group
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  // Retrieving MPI environment variables
  auto numranks = pg->getSize();
  auto rank = pg->getRank();

  // TRAINING
  // Read train dataset
  const char* kDataRoot = "../data";
  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  // Distributed Random Sampler
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(
      train_dataset.size().value(), numranks, rank, false);

  auto num_train_samples_per_proc = train_dataset.size().value() / numranks;

  // Generate dataloader
  auto total_batch_size = 64;
  auto batch_size_per_proc =
      total_batch_size / numranks; // effective batch size in each processor
  auto data_loader = torch::data::make_data_loader(
      std::move(train_dataset), data_sampler, batch_size_per_proc);

  // setting manual seed
  torch::manual_seed(0);

  auto model = std::make_shared<Model>();

  auto learning_rate = 1e-2;

  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  // Number of epochs
  size_t num_epochs = 10;

  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    size_t num_correct = 0;

    for (auto& batch : *data_loader) {
      auto ip = batch.data;
      auto op = batch.target.squeeze();

      // convert to required formats
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      // Reset gradients
      model->zero_grad();

      // Execute forward pass
      auto prediction = model->forward(ip);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      // Backpropagation
      loss.backward();

      // Averaging the gradients of the parameters in all the processors
      // Note: This may lag behind DistributedDataParallel (DDP) in performance
      // since this synchronizes parameters after backward pass while DDP
      // overlaps synchronizing parameters and computing gradients in backward
      // pass
      std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
      for (auto& param : model->named_parameters()) {
        std::vector<torch::Tensor> tmp = {param.value().grad()};
        auto work = pg->allreduce(tmp);
        works.push_back(std::move(work));
      }

      waitWork(pg, works);

      for (auto& param : model->named_parameters()) {
        param.value().grad().data() = param.value().grad().data() / numranks;
      }

      // Update parameters
      optimizer.step();

      auto guess = prediction.argmax(1);
      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
    } // end batch loader

    auto accuracy = 100.0 * num_correct / num_train_samples_per_proc;

    std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - "
              << accuracy << std::endl;

  } // end epoch

  // TESTING ONLY IN RANK 0
  if (rank == 0) {
    auto test_dataset =
        torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto num_test_samples = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset), num_test_samples);

    model->eval(); // enable eval mode to prevent backprop

    size_t num_correct = 0;

    for (auto& batch : *test_loader) {
      auto ip = batch.data;
      auto op = batch.target.squeeze();

      // convert to required format
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      auto prediction = model->forward(ip);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      std::cout << "Test loss - " << loss.item<float>() << std::endl;

      auto guess = prediction.argmax(1);

      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

    } // end test loader

    std::cout << "Num correct - " << num_correct << std::endl;
    std::cout << "Test Accuracy - " << 100.0 * num_correct / num_test_samples
              << std::endl;
  } // end rank 0
}
