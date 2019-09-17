//
//  LibTorchPredictor.m
//  PyTorchDemo
//
//  Created by Tao Xu on 9/16/19.
//

#import "LibTorchPredictor.h"
#import <LibTorch/LibTorch.h>

@implementation LibTorchPredictor {
    torch::jit::script::Module _moduleImpl;
}

- (BOOL)loadTorchScriptModel:(NSString*) modelPath {
     _moduleImpl = torch::jit::load([[[NSBundle mainBundle] pathForResource:@"model" ofType:@"pt"] cStringUsingEncoding:NSASCIIStringEncoding]);
    return true;
}

@end
