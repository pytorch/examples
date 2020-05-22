#include<torch/torch.h>
#include<torch/script.h>
#include<cstdio>
#include<iostream>
#include<float.h>
#include<fstream>
#include<sstream>
#include<string>
#include<regex>
#include<tuple>
#include<typeinfo>
#include<vector>
#define seed 1
#include "loader.h"
#include "datautils.h"
#include "model.h"
#include "trainmodel.h"

// Path to data
const std::string datapath = "../TREC/";

// Maximum length of sentence
const int32_t kMaxLen = 20;

// Embedding Dimensions
const int32_t kEmbedDim  = 300;

// Number of classes in TREC Dataset
const int32_t kNumClasses = 6;

const int32_t kNumfilters = 100;

int32_t kFilterSizes[] = {3, 4, 5};

double kDropValue = 0.5;

// Batch size
int32_t kBatchSize =  16;

// Number of epochs to train
int32_t kNumofEpochs = 10;

int main(int argc, char** argv) {
    torch::manual_seed(seed);

    std::vector<std::string> train_corpus;
    std::vector<int32_t> train_labels;
    std::pair< std::vector<std::string>, std::vector<std::int32_t> > ktrain;

    ktrain = loader::load_data(datapath, "train.txt");
    std::cout << "Loaded Training data" << std::endl;
    train_corpus = ktrain.first;
    train_labels = ktrain.second;

    std::vector<std::string> test_corpus;
    std::vector<int32_t> test_labels;
    std::pair< std::vector<std::string>, std::vector<std::int32_t> > ktest;
    ktest = loader::load_data(datapath, "test.txt");
    std::cout << "Loaded Testing data" << std::endl;
    test_corpus = ktest.first;
    test_labels = ktest.second;

    if (train_corpus.size() == train_labels.size()) {
        std::cout << "Training Data Size:" << " " << train_corpus.size();
        std:: cout << std::endl;
    }

    if (test_corpus.size() == test_labels.size()) {
        std::cout << "Test Data Size:" << " " << test_corpus.size();
        std::cout << std::endl;
    }

    std::map<std::string, int32_t> vocab;
    vocab = datautils::create_vocab(train_corpus);

    std::pair<torch::Tensor, torch::Tensor> trainindices_labels;
    std::pair<torch::Tensor, torch::Tensor> testindices_labels;

    // Converting Training Data into Training Data Loader
    trainindices_labels = datautils::get_loader(vocab, train_corpus,
        train_labels, kMaxLen, kBatchSize);

    auto traindata_set = datautils::CustomDataset(trainindices_labels.first,
        trainindices_labels.second).map(
             torch::data::transforms::Stack<>());

    auto train_dataset_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(traindata_set), kBatchSize);

    // Converting Testing Data into Test Data Loader
    testindices_labels = datautils::get_loader(vocab, test_corpus, test_labels,
        kMaxLen, kBatchSize);

    auto testdata_set = datautils::CustomDataset(testindices_labels.first,
        testindices_labels.second).map(
            torch::data::transforms::Stack<>());

    auto test_dataset_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(testdata_set), kBatchSize);

    int32_t vocabsize = vocab.size();

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "GPU available. Training on GPU." << std::endl;
        torch::Device device(torch::kCUDA);
    } else {
        std::cout << "No GPU available. Training on CPU. " << std::endl;
    }

    RandCNN randcnn(vocabsize, kEmbedDim, kMaxLen, kNumClasses, kNumfilters,
        kFilterSizes, kDropValue);
    randcnn.to(device);

    trainmodel::train_model(
        kNumofEpochs,
        randcnn,
        *train_dataset_loader,
        *test_dataset_loader,
        device);
}
