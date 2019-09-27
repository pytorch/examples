import UIKit

class NLPPredictor: Predictor {
    var module: TorchModule?
    var labels: [String] = []
    var isRunning: Bool = false

    init() {
        module = loadModel(name: "reddit")
        labels = loadLabels(name: "topics")
    }

    func forward(_ text: String, resultCount: Int, completionHandler: ([InferenceResult]?, Double, Error?) -> Void) {
        if isRunning {
            return
        }
        isRunning = true
        let startTime = CFAbsoluteTimeGetCurrent()
        guard let outputBuffer = module?.predictText(text) else {
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
