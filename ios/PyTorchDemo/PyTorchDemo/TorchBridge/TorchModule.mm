//
//  FBTorchModule.m
//  PyTorchDemo
//
//  Created by taox on 9/22/19.
//

#import "TorchModule.h"
#import <torch/script.h>

@implementation TorchModule {
@protected
  torch::jit::script::Module _impl;
}

+ (nullable instancetype)loadModel:(NSString*)modelPath {
    if (modelPath.length == 0) {
        return nil;
    }
    @try {
        TorchModule* module = [[self.class alloc]init];
        module->_impl = torch::jit::load([modelPath cStringUsingEncoding:NSASCIIStringEncoding]);
        return module;
    } @catch (NSException* exception) {
        @throw exception;
        NSLog(@"%@", exception);
    }
    return nil;
}
@end

@implementation TorchVisionModule

- (nullable const void* ) predict:(void*)imageBuffer tensorSizes:(NSArray<NSNumber* >* )sizes {
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < sizes.count; ++i) {
        int64_t dim = sizes[i].integerValue;
        dimsVec.push_back(dim);
    }
    @try {
        at::Tensor tensor = torch::from_blob((void*)imageBuffer, {1,3,224,224}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        auto outputTensor = _impl.forward({tensor}).toTensor();
        return outputTensor.unsafeGetTensorImpl()->data();
    } @catch (NSException* exception) {
        @throw exception;
        NSLog(@"%@", exception);
    }
    return nil;
}

@end
