#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

+ (nullable instancetype)loadModel:(NSString*)modelPath;
- (nullable const void*)predictImage:(void*)imageBuffer;
- (nullable const void*)predictText:(NSString*)text;

@end

NS_ASSUME_NONNULL_END
