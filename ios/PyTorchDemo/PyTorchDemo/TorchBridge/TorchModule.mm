//
//  FBTorchModule.m
//  PyTorchDemo
//
//  Created by taox on 9/22/19.
//

#import "TorchModule.h"
#import <torch/csrc/autograd/grad_mode.h>
#import <torch/script.h>

#define DEFINE_TENSOR_TYPES(_) \
  _(Byte)                      \
  _(Char)                      \
  _(Int)                       \
  _(Float)                     \
  _(Long)                      \
  _(Undefined)

static inline c10::ScalarType scalarTypeFromTensorType(TensorType type) {
  switch (type) {
#define DEFINE_CASE(x) \
  case TensorType##x:  \
    return c10::ScalarType::x;
    DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
  }
  return c10::ScalarType::Undefined;
}

@implementation TorchModule {
  torch::jit::script::Module _impl;
  at::Tensor _outputTensor;
}

+ (instancetype)sharedInstance {
  static dispatch_once_t onceToken = 0;
  static TorchModule* instance = nil;
  dispatch_once(&onceToken, ^{
    instance = [[TorchModule alloc] init];
  });
  return instance;
}

- (void*)data {
  return _outputTensor.unsafeGetTensorImpl()->data();
}

- (BOOL)loadModel:(NSString*)modelPath {
  if (modelPath.length == 0) {
    return NO;
  }
  @try {
    _impl = torch::jit::load([modelPath cStringUsingEncoding:NSASCIIStringEncoding]);
    return YES;
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return NO;
}

- (BOOL)predict:(void*)buffer tensorSizes:(NSArray<NSNumber*>*)sizes tensorType:(TensorType)type {
  if (!buffer) {
    return NO;
  }
  std::vector<int64_t> dimsVec;
  for (auto i = 0; i < sizes.count; ++i) {
    int64_t dim = sizes[i].integerValue;
    dimsVec.push_back(dim);
  }
  @try {
    at::Tensor tensor = torch::from_blob((void*)buffer, dimsVec, scalarTypeFromTensorType(type));
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    _outputTensor = _impl.forward({tensor}).toTensor();
    return YES;
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return NO;
}

@end
