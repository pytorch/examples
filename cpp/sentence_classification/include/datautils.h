#include<torch/torch.h>
#include<cstdio>
#include<iostream>
#include<string>
#include<vector>

namespace datautils {

//  Function to create vocabulary
std::map<std::string, int32_t> create_vocab(std::vector<std::string>);

//  Function to return tensor of indices for a sentence with padding
torch::Tensor get_indices(
                        std::string,
                        std::map<std::string,
                        int32_t>,
                        int32_t);

//  Function to return dataloader with sentences converted to
//  vocabulary indices and labels converted to tensors
std::pair<torch::Tensor, torch::Tensor> get_loader(
    std::map<std::string, int32_t> &dic,
    std::vector<std::string> &sen,
    std::vector<int32_t> &labels,
    int32_t,
    int32_t);

//  Class to convert indices, labels into DataLoader
class CustomDataset : public torch::data::Dataset<CustomDataset> {
 private:
        torch::Tensor inputs_, targets_;
 public:
        CustomDataset(torch::Tensor emb_indices, torch::Tensor labels) {
            inputs_ = emb_indices;
            targets_ = labels;
         }
         torch::data::Example<> get(size_t index) override {
            torch::Tensor embedding_index = inputs_[index];
            torch::Tensor Label = targets_[index];
            return {embedding_index.clone(), Label.clone()};
         };
         torch::optional<size_t> size() const override{
            return at::size(inputs_, 0);
         };
};
}  // namespace datautils
