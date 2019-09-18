//
//  Data+Helper.swift
//  PyTorchDemo
//
//  Created by Tao Xu on 9/17/19.
//

import Foundation

extension Data {
    init<T>(fromArray array:[T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    
}
