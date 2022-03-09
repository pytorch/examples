//
//  main.h
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
//

#ifndef main_h
#define main_h

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <dirent.h>
#include <torch/script.h>

// Function to return image read at location given as type torch::Tensor
// Resizes image to (224, 224, 3)
torch::Tensor read_data(std::string location);

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
torch::Tensor read_label(int label);

// Function returns vector of tensors (images) read from the list of images in a folder
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images);

// Function returns vector of tensors (labels) read from the list of labels
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

// Function to load data from given folder(s) name(s) (folders_name)
// Returns pair of vectors of string (image locations) and int (respective labels)
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

// Function to train the network on train data
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

// Custom Dataset class
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states = process_images(list_images);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};

#endif /* main_h */
