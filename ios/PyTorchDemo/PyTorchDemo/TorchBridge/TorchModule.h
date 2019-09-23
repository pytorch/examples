//
//  FBTorchModule.h
//  PyTorchDemo
//
//  Created by taox on 9/22/19.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, TensorType) {
  TensorTypeByte,       // 8bit unsigned integer
  TensorTypeChar,       // 8bit signed integer
  TensorTypeInt,        // 32bit signed integer
  TensorTypeLong,       // 64bit signed integer
  TensorTypeFloat,      // 32bit single precision floating point
  TensorTypeUndefined,  // Undefined tensor type. This indicates an error with the model
};

@interface TorchModule : NSObject

@property(nonatomic, readonly, nullable) void* data;

+ (instancetype)sharedInstance;

- (BOOL)loadModel:(NSString*)modelPath;

- (BOOL)predict:(void*)buffer tensorSizes:(NSArray<NSNumber*>*)sizes tensorType:(TensorType)type;

@end

NS_ASSUME_NONNULL_END
