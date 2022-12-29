#ifndef model
#define model
#include<torch/torch.h>

using namespace torch;
using namespace at;

struct RandCNN : nn::Module {
  RandCNN(int32_t vocabsize, int32_t embeddim, int32_t maxlen,
        int32_t numclasses, int32_t kNumfilters, int32_t kFilterSizes[],
        double kDropValue)
      : embed(nn::Embedding(nn::EmbeddingOptions(vocabsize, 
                                                embeddim))),
        conv1(nn::Conv1d(nn::Conv1dOptions(embeddim, kNumfilters, kFilterSizes[0])
                            .stride(1)
                            .bias(false))),
        conv2(nn::Conv1d(nn::Conv1dOptions(embeddim, kNumfilters, kFilterSizes[1])
                            .stride(1)
                            .bias(false))),
        conv3(nn::Conv1d(nn::Conv1dOptions(embeddim, kNumfilters, kFilterSizes[2])
                            .stride(1)
                            .bias(false))),
        pool1(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[0] + 1)
                               .stride(1))),
        pool2(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[1] + 1)
                               .stride(1))),
        pool3(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[2] + 1)
                               .stride(1))),
        fc(nn::Linear(nn::LinearOptions(kNumfilters * 3, numclasses))),
        drop(nn::Dropout(nn::DropoutOptions().p(kDropValue))) {

   // register_module() is needed if we want to use the parameters() method later on
    register_module("embed", embed);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("poo11", pool1);
    register_module("pool2", pool2);
    register_module("pool3", pool3);
    register_module("fc", fc);
}

 torch::Tensor forward(torch::Tensor x) {
    x = embed(x);
    x = at::transpose(x, 1, 2);
    torch::Tensor out1, out2, out3, out;
    out1 = torch::relu(conv1(x));
    out2 = torch::relu(conv2(x));
    out3 = torch::relu(conv3(x));
    out1 = pool1(out1);
    out2 = pool2(out2);
    out3 = pool3(out3);
    out = at::cat({out1, out2, out3}, 1);
    out = at::_unsafe_view(out, {at::size(out, 0), at::size(out, 1)});
    out = drop(out);
    out = fc(out);
   return out;
 }

 nn::Embedding embed;
 nn::Conv1d conv1, conv2, conv3;
 nn::MaxPool1d pool1, pool2, pool3;
 nn::Linear fc;
 nn::Dropout drop;
};
#endif
