#import "TorchModule.h"
#import <Foundation/Foundation.h>
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
    TorchModule* module = [[self.class alloc] init];
    at::globalContext().setQEngine(at::QEngine::QNNPACK);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    module->_impl = torch::jit::load(modelPath.UTF8String);
    module->_impl.eval();
    return module;
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

- (nullable const void*)predictImage:(void*)imageBuffer {
  @try {
    at::Tensor tensor = torch::from_blob((void*)imageBuffer, {1, 3, 224, 224}, at::kFloat);
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

- (nullable const void*)predictText:(NSString*)text {
  @try {
    uint8_t* buffer = (uint8_t*)text.UTF8String;
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::Tensor tensor = torch::from_blob((void*)buffer, {1, static_cast<long long>(text.length)}, at::kByte);
    auto outputTensor = _impl.forward({tensor}).toTensor();
    return outputTensor.unsafeGetTensorImpl()->data();
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

@end
