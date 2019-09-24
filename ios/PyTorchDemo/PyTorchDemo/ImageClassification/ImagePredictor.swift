import UIKit

struct InferenceResult {
    let score: Float32
    let label: String
}

enum ModelContext {
    static let model = (name: "ResNet18", type: "pt")
    static let label = (name: "Labels", type: "txt")
    static let inputTensorSize  = [1, 3, 224, 224]
    static let outputTensorSize = [1, 1000]
}

class ImagePredictor: NSObject {
    private var module: TorchVisionModule?
    private var labels: [String] = []
    private var isRunning = false

    override init() {
        super.init()
        module = loadModel()
        labels = loadLabels()
    }

    func forward(_ buffer: [Float32]?, completionHandler: ([InferenceResult]?, Double, Error?) -> Void) {
        guard var tensorBuffer = buffer else {
            return
        }
        if isRunning {
            return
        }
        isRunning = true
        let startTime = CFAbsoluteTimeGetCurrent()
        guard let outputBuffer = module?.predict(UnsafeMutableRawPointer(&tensorBuffer), tensorSizes: ModelContext.inputTensorSize as [NSNumber]) else {
            completionHandler([], 0.0, ImagePredictorError.invalidInputTensor)
            return
        }
        let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime)*1000
        let outputs = outputBuffer.floatArray(size: ModelContext.outputTensorSize[1])
        let results = getTopN(scores: outputs, count: 3, inferenceTime: inferenceTime)
        completionHandler(results, inferenceTime, nil)
        isRunning = false
    }

    private func getTopN(scores: [Float32], count: Int, inferenceTime: Double) -> [InferenceResult] {
        let zippedResults = zip(labels.indices, scores)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(count)
        return sortedResults.map { InferenceResult(score: $0.1, label: labels[$0.0]) }
    }

    private func loadLabels() -> [String] {
        if let filePath = Bundle.main.path(forResource: ModelContext.label.name, ofType: ModelContext.label.type),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }

    private func loadModel() -> TorchVisionModule? {
        if let filePath = Bundle.main.path(forResource: ModelContext.model.name, ofType: ModelContext.model.type),
            let module = TorchVisionModule.loadModel(filePath){
            return module
        } else {
            fatalError("Can't find the model with the given path!")
        }
    }
}

extension ImagePredictor {
    enum ImagePredictorError: Swift.Error {
        case invalidModel
        case invalidInputTensor
        case invalidOutputTensor
    }
}
