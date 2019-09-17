#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, TorchTensorType) {
    TorchTensorTypeByte, //8bit unsigned integer
    TorchTensorTypeInt,  //32bit signed integer
    TorchTensorTypeLong, //64bit signed integer
    TorchTensorTypeFloat, //32bit single precision floating point
    TorchTensorTypeUndefined, //Undefined tensor type. This indicates an error with the model
};
/**
 An input or output tensor model
 */
@interface TorchTensor : NSObject<NSCopying>
/**
 Data type of the tensor
 */
@property(nonatomic,assign, readonly) TorchTensorType type;
/**
//The size of the tensor. The returned value is a array of integer
 */
@property(nonatomic,strong, readonly) NSArray<NSNumber* >* sizes;
/**
 /The number of dimensions of the tensor.
 */
@property(nonatomic,assign, readonly) int64_t dim;
/**
 The total number of elements in the input tensor.
 */
@property(nonatomic,assign, readonly) int64_t numel;
/**
 Returns if the tensor has a quntized backend
 */
@property(nonatomic,assign, readonly) BOOL quantized;
/**
 Creat a tensor object with data type, shape and a raw pointer to a data buffer.

 @param type Data type of the tensor
 @param size Size of the tensor
 @param data A raw pointer to a data buffer
 @return  A tensor object
 */
+ (nullable TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data;

@end

@interface TorchTensor(Operations)
/**
 Performs Tensor dtype and/or device conversion.

 @param type Data type
 @return A new tensor with the given type
 */
- (nullable TorchTensor* )to:(TorchTensorType) type;

@end

@interface TorchTensor(ObjectSubscripting)
/**
 This allows the tensor obejct to do subscripting. For example, let's say the shape of the current tensor is `1x10`,
 then tensor[0] will give you a one dimentional array of 10 tensors.

 @param idx index
 @return A new tensor object with the given index
 */
- (nullable TorchTensor* )objectAtIndexedSubscript:(NSUInteger)idx;

@end

NS_ASSUME_NONNULL_END
