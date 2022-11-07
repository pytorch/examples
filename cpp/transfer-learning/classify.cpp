//
//  classify.cpp
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
//

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <dirent.h>

// Utility function to load image from given folder
// File type accepted: .jpg
std::vector<std::string> load_images(std::string folder_name) {
    std::vector<std::string> list_images;
    std::string base_name = folder_name;
    DIR* dir;
    struct dirent *ent;
    if((dir = opendir(base_name.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                std::string newf = base_name + filename;
                list_images.push_back(newf);
            }
        }
    }
    return list_images;
}

void print_probabilities(std::string loc, std::string model_path, std::string model_path_linear) {
    // Load image with OpenCV.
    cv::Mat img = cv::imread(loc);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);
    
    // Load the model.
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);
    
    torch::nn::Linear model_linear(512, 2);
    torch::load(model_linear, model_path_linear);
    
    // Predict the probabilities for the classes.
    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor prob = model.forward(input).toTensor();
    prob = prob.view({prob.size(0), -1});
    prob = model_linear(prob);
    
    std::cout << "Printing for location: " << loc << std::endl;
    std::cout << "Cat prob: " << *(prob.data<float>())*100. << std::endl;
    std::cout << "Dog prob: " << *(prob.data<float>()+1)*100. << std::endl;
}

int main(int arc, char** argv)
{
    // argv[1] should is the test image
    std::string location = argv[1];
    
    // argv[2] contains pre-trained model without last layer
    // argv[3] contains trained last FC layer
    std::string model_path = argv[2];
    std::string model_path_linear = argv[3];
    
    // Load the model.
    // You can also use: auto model = torch::jit::load(model_path);
    torch::jit::script::Module model = torch::jit::load(model_path);
    
    // Print probabilities for dog and cat classes
    print_probabilities(location, model_path, model_path_linear);
    return 0;
}
