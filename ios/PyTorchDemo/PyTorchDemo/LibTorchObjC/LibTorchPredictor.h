//
//  LibTorchPredictor.h
//  PyTorchDemo
//
//  Created by Tao Xu on 9/16/19.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LibTorchPredictor : NSObject

- (BOOL)loadTorchScriptModel:(nonnull NSString*) modelPath;


@end

NS_ASSUME_NONNULL_END
