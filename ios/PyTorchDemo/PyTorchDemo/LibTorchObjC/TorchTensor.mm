#import "TorchTensor.h"
#import "TorchTensor+Internal.h"
#import <LibTorch/LibTorch.h>

#define CHECK_IMPL(x) NSCAssert(x!=nil,@"impl is nil!");
#define CHECK_IMPL_(x) \
    CHECK_IMPL(x) \
    if (!x) { return nil; }

#define DEFINE_TENSOR_TYPES(_) \
    _(Byte) \
    _(Int) \
    _(Float) \
    _(Long) \
    _(Undefined)

#define DEFINE_TENSOR_SCALAR_TYPES(_) \
    _(Int) \
    _(Float) \
    _(Long) \

static inline c10::ScalarType scalarTypeFromTensorType(TorchTensorType type) {
    switch(type){
#define DEFINE_CASE(x) case TorchTensorType##x: return c10::ScalarType::x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    }
    return c10::ScalarType::Undefined;
}

static inline TorchTensorType tensorTypeFromScalarType(c10::ScalarType type) {
    switch(type){
#define DEFINE_CASE(x) case c10::ScalarType::x: return TorchTensorType##x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default: return TorchTensorTypeUndefined;
    }
}

@implementation TorchTensor {
    at::Tensor _impl;
}

- (TorchTensorType)type {
    return tensorTypeFromScalarType(_impl.scalar_type());
}

- (NSArray<NSNumber* >* )sizes {
    NSMutableArray* shapes = [NSMutableArray new];
    auto dims = _impl.sizes();
    for (int i=0; i<dims.size(); ++i){
        [shapes addObject:@(dims[i])];
    }
    return [shapes copy];
}

- (int64_t)numel{
    return _impl.numel();
}

- (void* )data {
    return _impl.unsafeGetTensorImpl()->storage().data();
}

- (int64_t)dim {
    return _impl.dim();
}


+ (TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data {
    if (!data || size.count == 0){
        return nil;
    }
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < size.count; ++i) {
        int64_t dim = size[i].integerValue;
        dimsVec.push_back(dim);
    }
    at::Tensor tensor = torch::from_blob( (void* )data, dimsVec,scalarTypeFromTensorType(type));
    return [TorchTensor newWithTensor:tensor];
}

- (NSString* )description {
    NSString* size = @"[";
    for(NSNumber* num in self.sizes) {
        size = [size stringByAppendingString:[NSString stringWithFormat:@"%ld", num.integerValue]];
    }
    size = [size stringByAppendingString:@"]"];
    return [NSString stringWithFormat:@"[%s %@]",_impl.toString().c_str(),size];
}

- (TorchTensor* )objectAtIndexedSubscript:(NSUInteger)idx {
    auto tensor = _impl[idx];
    return [TorchTensor newWithTensor:tensor];
}

- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx {
    NSAssert(NO, @"Tensors are immutable");
}

#pragma mark NSCopying

- (instancetype)copyWithZone:(NSZone *)zone {
    //tensors are immutable
    return self;
}

@end

@implementation TorchTensor (Internal)

- (at::Tensor)toTensor {
    return at::Tensor(_impl);
}

+ (TorchTensor* )newWithTensor:(const at::Tensor& ) tensor{
    TorchTensor* t = [TorchTensor new];
    //TODO: trigger copy constructor?
    t->_impl = tensor;
    return t;
}

@end

@implementation TorchTensor (Operations)

- (NSNumber* )item {
    if( self.numel > 1 ){
        return nil;
    }
    switch (self.type) {
#define DEFINE_CASE(x) case TorchTensorType##x: return @(_impl.item().to##x());
            DEFINE_TENSOR_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default:
            return nil;
    }
}
@end
