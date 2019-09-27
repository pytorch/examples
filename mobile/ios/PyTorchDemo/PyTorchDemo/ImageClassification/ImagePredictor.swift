import UIKit

class ImagePredictor: Predictor {
    var isRunning: Bool = false
    var module: TorchModule?
    var labels: [String] = []

    init() {
        module = loadModel(name: "mobilenet_quantized")
        labels = loadLabels(name: "words")
    }

    func forward(_ buffer: [Float32]?, resultCount: Int, completionHandler: ([InferenceResult]?, Double, Error?) -> Void) {
        guard var tensorBuffer = buffer else {
            return
        }
        if isRunning {
            return
        }
        isRunning = true
        let startTime = CFAbsoluteTimeGetCurrent()
        guard let outputBuffer = module?.predictImage(UnsafeMutableRawPointer(&tensorBuffer)) else {
            completionHandler([], 0.0, PredictorError.invalidInputTensor)
            return
        }
        let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        let outputs = outputBuffer.floatArray(size: labels.count)
        let results = getTopN(scores: outputs, count: resultCount, inferenceTime: inferenceTime)
        completionHandler(results, inferenceTime, nil)
        isRunning = false
    }
}
