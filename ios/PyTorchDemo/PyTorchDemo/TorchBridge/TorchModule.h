//
//  FBTorchModule.h
//  PyTorchDemo
//
//  Created by taox on 9/22/19.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

+ (nullable instancetype)loadModel:(NSString* )modelPath;

@end

@interface TorchVisionModule: TorchModule

- (nullable const void* ) predict:(void*)imageBuffer tensorSizes:(NSArray<NSNumber* >* )sizes;

@end

@interface TorchNLPModule: TorchModule
@end


NS_ASSUME_NONNULL_END
