#import "TorchTensor.h"
#import <LibTorch/LibTorch.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchTensor (Internal)

- (at::Tensor)toTensor;
+ (TorchTensor* )newWithTensor:(const at::Tensor& ) tensor;

@end

NS_ASSUME_NONNULL_END
