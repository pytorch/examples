import Foundation

extension UnsafeRawPointer {
    func floatArray(size: Int) -> [Float32] {
        let floatPtr = bindMemory(to: Float32.self, capacity: Int(size))
        var buffer: [Float32] = []
        for index in 0 ..< size {
            let value = floatPtr[Int(index)]
            buffer.append(value)
        }
        return buffer
    }
}
