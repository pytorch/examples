import Foundation

extension TorchTensor {
    func floatArray() -> [Float32] {
        let rawPtr: UnsafeMutableRawPointer = data
        let floatPtr = rawPtr.bindMemory(to: Float32.self, capacity: Int(numel))
        var buffer: [Float32] = []
        for index in 0 ..< numel {
            let value = floatPtr[Int(index)]
            buffer.append(value)
        }
        return buffer
    }
}
