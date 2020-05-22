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

namespace loader {
// Function to preprocess text
std::string preprocess(std::string &s) {
    std::regex reg1("(^A-Za-z0-9(),!?\'\\`)");
    s = std::regex_replace(s, reg1, " ");
    std::regex reg2("\'s");
    s = std::regex_replace(s, reg2, " \'s");
    std::regex reg3("\'ve");
    s = std::regex_replace(s, reg3, " \'ve");
    std::regex reg4("n\'t");
    s = std::regex_replace(s, reg4, " n\'t");
    std::regex reg5("\'re");
    s = std::regex_replace(s, reg5, " \'re");
    std::regex reg6("\'d");
    s = std::regex_replace(s, reg6, " \'d");
    std::regex reg7("\'ll");
    s = std::regex_replace(s, reg7, " \'ll");
    std::regex reg8(",");
    s = std::regex_replace(s, reg8, " , ");
    std::regex reg9("!");
    s = std::regex_replace(s, reg9, " ! ");
    std::regex reg10("\\(");
    s = std::regex_replace(s, reg10, " \\( ");
    std::regex reg11("\\)");
    s = std::regex_replace(s, reg11, " \\) ");
    std::regex reg12("\\?");
    s = std::regex_replace(s, reg12, " \\? ");
    std::regex reg13("\\s{2,}");
    s = std::regex_replace(s, reg13, " ");
    return s;
}

// Function to return sentences and labels
std::pair< std::vector<std::string>, std::vector<int32_t> > load_data(
    const std::string datapath,
    const std::string name) {
    std::string file_name = datapath + name;

    //  Map of strings and their corresponsing labels
    std::map<std::string, int32_t> categories;
    categories.insert(std::make_pair("NUM", 0));
    categories.insert(std::make_pair("LOC", 1));
    categories.insert(std::make_pair("ABBR", 2));
    categories.insert(std::make_pair("HUM", 3));
    categories.insert(std::make_pair("DESC", 4));
    categories.insert(std::make_pair("ENTY", 5));

    //  File to load
    std::ifstream file(file_name);

    //  Vector of Sentences
    std::vector<std::string> sentences;

    //  Vector of labels
    std::vector<int32_t> labels;

    std::string line, current_sen;
    while (std::getline(file, line)) {
        // Split by :
        auto pos = line.find(":");
        auto itr = categories.find(line.substr(0, pos));

        // Take sentences starting after :
        current_sen = line.substr(pos + 1);

        //  Ignore first word as it is also included in label
        pos = current_sen.find_first_of(" \t") + 1;
        current_sen = current_sen.substr(pos);

        //  Preprocess the sentence
        current_sen = preprocess(current_sen);
        labels.push_back(itr->second);
        sentences.push_back(current_sen);
    }
    file.close();
    return std::make_pair(sentences, labels);
}
}  // namespace loader
