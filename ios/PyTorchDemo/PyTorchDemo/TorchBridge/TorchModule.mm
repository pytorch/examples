//
//  FBTorchModule.m
//  PyTorchDemo
//
//  Created by taox on 9/22/19.
//

#import "TorchModule.h"
#import <torch/script.h>
#import <Foundation/Foundation.h>

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
- (nullable const void* ) predictImage:(void*)imageBuffer {
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

- (nullable const void* ) predictText:(NSString* )text {
    
    uint8_t* buffer = (uint8_t* )text.UTF8String;
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::Tensor tensor = torch::from_blob((void* )buffer, {1,static_cast<long long>(text.length)}, at::kByte);
    auto outputTensor = _impl.forward({tensor}).toTensor();
        return outputTensor.unsafeGetTensorImpl()->data();

}

@end
