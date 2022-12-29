#include<torch/torch.h>
#include<float.h>
#include<iomanip>
#include "model.h"

using namespace at;
namespace F = torch::nn::functional;

namespace trainmodel {
template <typename DataLoader>
std::pair<double, double> evaluate(
    auto& net,
    DataLoader& loader_,
    torch::Device device);

template <typename DataLoader>
std::pair<double, double> evaluate(
    auto& net,
    DataLoader& testloader_,
    torch::Device device) {
    torch::NoGradGuard no_grad;
    net.eval();
    double test_loss = 0.0;
    double correct = 0.0;
    double totalsize = 0.0;
    auto options_ = F::CrossEntropyFuncOptions().reduction(torch::kMean);
    for (auto& batch : testloader_) {
        auto inputs_ = batch.data.to(torch::kLong).to(device);
        auto targets_ = batch.target.to(torch::kLong).to(device);
        totalsize = totalsize + inputs_.size(0);
        auto output = net.forward(inputs_);
        auto loss = F::cross_entropy(output, targets_, options_);
        test_loss += loss.template item<double>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets_).sum().template item<double>();
    }
    test_loss = test_loss / totalsize;
    auto accuracy_ = correct / totalsize;
    return std::make_pair(test_loss, accuracy_);
}

template <typename DataLoader>
void train_model(
    size_t epochs,
    auto& Net,
    DataLoader& trainloader,
    DataLoader& testloader,
    torch::Device device) {
    torch::optim::Adam optimizer(
        Net.parameters(), torch::optim::AdamOptions(0.001));

    double train_loss = 0.0;
    double totalsize = 0.0;
    double test_loss, test_acc;
    auto options_ = F::CrossEntropyFuncOptions().reduction(torch::kMean);
    std::pair<double, double> values;
    for (size_t epoch = 1; epoch <= epochs; epoch++) {
            Net.train();
            train_loss = 0.0;
            for (auto& batch : trainloader) {
                auto inputs_ = batch.data.to(torch::kLong).to(device);
                auto targets_ = batch.target.to(torch::kLong).to(device);
                totalsize += 1;

                optimizer.zero_grad();
                auto output = Net.forward(inputs_);

                auto loss = F::cross_entropy(output, targets_, options_);
                loss.backward();
                torch::nn::utils::clip_grad_norm_(Net.parameters(), 3.0);
                optimizer.step();

                train_loss += loss.template item<double>();
            }
            train_loss = train_loss/totalsize;
            std::cout << "Epoch: " << epoch << " " << "Training Loss: ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3) << train_loss << std::endl;
    }
    values = evaluate(Net, testloader, device);
    test_loss = values.first;
    test_acc = values.second;
    std::cout << "Test Loss: " << test_loss << " " << "Test Accuracy: ";
    std::cout << test_acc << std::endl;
}
}  // namespace trainmodel
