#include<torch/torch.h>
#include<torch/script.h>
#include<chrono>
#include<cstdio>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<regex>
#include<tuple>
#include<typeinfo>
#include<vector>

#include "datautils.h"

// Compare function to sort vocabulary by frequency.
bool compare(const std::pair<std::string, int32_t> &a,
             const std::pair<std::string, int32_t> &b) {
    return a.second > b.second;
}

// Function to return vocabulary
std::map<std::string, int32_t> datautils::create_vocab(
    std::vector<std::string> corpus) {
    //  Create a Map with words and frequencies
    std::map<std::string, int32_t> vocab;
    std::string word;

    for (size_t i = 0; i < corpus.size(); i++) {
        std::stringstream cursen(corpus[i]);
        while (cursen >> word) {
            auto index = vocab.find(word);
            //  If word is nor present add word to map
            // with frequency 1
            if (index == vocab.end())
                vocab.insert(std::make_pair(word, 1));
            // If word is found add 1 to its frequency
            else
                index -> second += 1;
        }
    }

    // Copy map to vector to sort according to frequency.
    std::vector<std::pair<std::string, int32_t> > vocabcopy;
    for (auto itr = vocab.begin(); itr != vocab.end(); itr++)
        vocabcopy.push_back(std::make_pair(itr -> first, itr -> second));

    // Sort words according to frequency.
    sort(vocabcopy.begin(), vocabcopy.end(), compare);

    // copy back sorted vector to dictionary map.
    std::map<std::string, int32_t> dictionary;

    // Add Padding index 0 to the dictionary map.
    dictionary.insert(std::make_pair("<PAD>", 0));

    for (size_t i = 0; i < vocabcopy.size(); i++) {
        dictionary.insert(std::make_pair(vocabcopy[i].first, i+1));
    }

    return dictionary;
}

// Function to return tensor of indices
torch::Tensor datautils::get_indices(
    std::string sentence,
    std::map<std::string, int32_t> dictionary, int32_t kMaxLen) {
    std::stringstream cursen(sentence);
    std::string word;

    // Initialize sentence with indices of 0's
    torch::Tensor embedding_indices = torch::zeros({kMaxLen});

    //  Calculate current sentence length
    int32_t sentence_length = sentence.length();

    //  startindex and endindex are needed to pad sentence
    int32_t startindex, endindex;

    //  If sentence length is less than kMaxLen
    //  startndex will be the difference between them
    //  endindex will be till length of sentence
    //  Else startindex will be 0 and endindex
    //  will be till kMaxLen
    if (sentence_length <= kMaxLen) {
        startindex = kMaxLen - sentence_length;
        endindex = sentence_length;
    } else {
        startindex = 0;
        endindex = kMaxLen;
    }

    //  Loop through each word in the sentence
    while ((cursen >> word) && (startindex < endindex)) {
        //  Find the index of word in dictionary
        auto position = dictionary.find(word);

        //  If word id found assign the index
        if (position != dictionary.end())
            embedding_indices[startindex] = position -> second;
        startindex++;
    }
    return embedding_indices;
}

std::pair<torch::Tensor, torch::Tensor> datautils::get_loader(
    std::map<std::string, int32_t> &dictionary,
    std::vector<std::string> &sentences,
    std::vector<int32_t> &labels,
    int32_t kMaxLen,
    int32_t kBatchSize
    ) {
    int32_t datasize = sentences.size();

    //  Tensor of indices with lenght kMaxLen
    torch::Tensor tensorindices = torch::zeros({datasize, kMaxLen});

    // For each sentence convert words into indices as in vocabulary
    for (size_t i = 0; i < sentences.size(); i++)
        tensorindices[i] = datautils::get_indices(
                                                sentences[i],
                                                dictionary,
                                                kMaxLen);

    auto options = torch::TensorOptions().dtype(torch::kInt32);

    //  Convert vector of indices to Tensor
    auto tensorlabels = torch::from_blob(labels.data(), {datasize}, options);
    return std::make_pair(tensorindices, tensorlabels);
}
