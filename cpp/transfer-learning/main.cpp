//
//  main.cpp
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 12/08/19.
//

#include "main.h"

torch::Tensor read_data(std::string location) {
    /*
     Function to return image read at location given as type torch::Tensor
     Resizes image to (224, 224, 3)
     Parameters
     ===========
     1. location (std::string type) - required to load image from the location
     
     Returns
     ===========
     torch::Tensor type - image read as tensor
    */
    cv::Mat img = cv::imread(location, 1);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

torch::Tensor read_label(int label) {
    /*
     Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
     Parameters
     ===========
     1. label (int type) - required to convert int to tensor
     
     Returns
     ===========
     torch::Tensor type - label read as tensor
    */
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    /*
     Function returns vector of tensors (images) read from the list of images in a folder
     Parameters
     ===========
     1. list_images (std::vector<std::string> type) - list of image paths in a folder to be read
     
     Returns
     ===========
     std::vector<torch::Tensor> type - Images read as tensors
     */
    std::vector<torch::Tensor> states;
    for(std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it);
        states.push_back(img);
    }
    return states;
}

std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    /*
     Function returns vector of tensors (labels) read from the list of labels
     Parameters
     ===========
     1. list_labels (std::vector<int> list_labels) -
     
     Returns
     ===========
     std::vector<torch::Tensor> type - returns vector of tensors (labels)
     */
    std::vector<torch::Tensor> labels;
    for(std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

std::pair<std::vector<std::string>,std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name) {
    /*
     Function to load data from given folder(s) name(s) (folders_name)
     Returns pair of vectors of string (image locations) and int (respective labels)
     Parameters
     ===========
     1. folders_name (std::vector<std::string> type) - name of folders as a vector to load data from
     
     Returns
     ===========
     std::pair<std::vector<std::string>, std::vector<int>> type - returns pair of vector of strings (image paths) and respective labels' vector (int label)
     */
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    int label = 0;
    for(auto const& value: folders_name) {
        std::string base_name = value + "/";
        // cout << "Reading from: " << base_name << endl;
        DIR* dir;
        struct dirent *ent;
        if((dir = opendir(base_name.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                    // cout << base_name + ent->d_name << endl;
                    // cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
                    list_images.push_back(base_name + ent->d_name);
                    list_labels.push_back(label);
                }
            }
            closedir(dir);
        } else {
            std::cout << "Could not open directory" << std::endl;
            // return EXIT_FAILURE;
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}

template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size) {
    /*
     This function trains the network on our data loader using optimizer.
     
     Also saves the model as model.pt after every epoch.
     Parameters
     ===========
     1. net (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. data_loader (DataLoader& type) - Training data loader
     4. optimizer (torch::optim::Optimizer& type) - Optimizer like Adam, SGD etc.
     5. size_t (dataset_size type) - Size of training dataset
     
     Returns
     ===========
     Nothing (void)
     */
    float best_accuracy = 0.0; 
    int batch_index = 0;
    
    for(int i=0; i<25; i++) {
        float mse = 0;
        float Acc = 0.0;
        
        for(auto& batch: *data_loader) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            
            // Should be of length: batch_size
            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);
            
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();
            
            auto output = net.forward(input).toTensor();
            // For transfer learning
            output = output.view({output.size(0), -1});
            output = lin(output);
            
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
            
            loss.backward();
            optimizer.step();
            
            auto acc = output.argmax(1).eq(target).sum();
            
            Acc += acc.template item<float>();
            mse += loss.template item<float>();
            
            batch_index += 1;
        }

        mse = mse/float(batch_index); // Take mean of loss
        std::cout << "Epoch: " << i  << ", " << "Accuracy: " << Acc/dataset_size << ", " << "MSE: " << mse << std::endl;

        test(net, lin, data_loader, dataset_size);

        if(Acc/dataset_size > best_accuracy) {
            best_accuracy = Acc/dataset_size;
            std::cout << "Saving model" << std::endl;
            net.save("model.pt");
            torch::save(lin, "model_linear.pt");
        }
    }
}

template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size) {
    /*
     Function to test the network on test data
     
     Parameters
     ===========
     1. network (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. loader (Dataloader& type) - test data loader
     4. data_size (size_t type) - test data size
     
     Returns
     ===========
     Nothing (void)
     */
    network.eval();
    
    float Loss = 0, Acc = 0;
    
    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();
        
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        auto output = network.forward(input).toTensor();
        output = output.view({output.size(0), -1});
        output = lin(output);
        
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    
    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main(int argc, const char * argv[]) {
    // Set folder names for cat and dog images
    std::string cats_name = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/cat_test";
    std::string dogs_name = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/dog_test";
    
    std::vector<std::string> folders_name;
    folders_name.push_back(cats_name);
    folders_name.push_back(dogs_name);
    
    // Get paths of images and labels as int from the folder paths
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folders_name);
    
    std::vector<std::string> list_images = pair_images_labels.first;
    std::vector<int> list_labels = pair_images_labels.second;
    
    // Initialize CustomDataset class and read data
    auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());

    // Load pre-trained model
    // You can also use: auto module = torch::jit::load(argv[1]);
    torch::jit::script::Module module = torch::jit::load(argv[1]);
    
    // Resource: https://discuss.pytorch.org/t/how-to-load-the-prebuilt-resnet-models-or-any-other-prebuilt-models/40269/8
    // For VGG: 512 * 14 * 14, 2

    torch::nn::Linear lin(512, 2); // the last layer of resnet, which we want to replace, has dimensions 512x1000
    torch::optim::Adam opt(lin->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), 4);

    train(module, lin, data_loader, opt, custom_dataset.size().value());
    return 0;
}
