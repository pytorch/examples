#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class TorchTensor;

@interface TorchIValue : NSObject

+ (instancetype)newWithTensor:(TorchTensor*)tensor;
+ (instancetype)newWithBoolValue:(bool)value;
+ (instancetype)newWithDoubleValue:(double)value;
+ (instancetype)newWithIntValue:(int64_t)value;
+ (instancetype)newWithBoolList:(NSArray<NSNumber*>*)value;
+ (instancetype)newWithIntList:(NSArray<NSNumber*>*)value;
+ (instancetype)newWithDoubleList:(NSArray<NSNumber*>*)value;
+ (instancetype)newWithTensorList:(NSArray<TorchTensor*>*)value;

- (instancetype)init NS_UNAVAILABLE;

- (BOOL)isTensor;
- (BOOL)isBool;
- (BOOL)isDouble;
- (BOOL)isInt;
- (BOOL)isBoolList;
- (BOOL)isDoubleList;
- (BOOL)isIntList;
- (BOOL)isTensorList;

- (TorchTensor*)toTensor;
- (bool)toBool;
- (int64_t)toInt;
- (double)toDouble;
- (NSArray<NSNumber*>*)toBoolList;
- (NSArray<NSNumber*>*)toIntList;
- (NSArray<NSNumber*>*)toDoubleList;
- (NSArray<TorchTensor*>*)toTensorList;

@end

NS_ASSUME_NONNULL_END
