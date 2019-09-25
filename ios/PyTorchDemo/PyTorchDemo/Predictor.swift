import UIKit

struct InferenceResult {
    let score: Float32
    let label: String
}

struct ModelContext {
    let model: (name: String, type: String)
    let label: (name: String, type: String)
    let inputTensorSize: [UInt]
    let outputTensorSize: [UInt]
}

enum VisionModelContext {
    static let model = (name: "ResNet18", type: "pt")
    static let label = (name: "Labels", type: "txt")
    static let inputTensorSize  = [1, 3, 224, 224]
    static let outputTensorSize = [1, 1000]
}

class Predictor: NSObject  {
    private var module: TorchModule?
    private var labels: [String] = []
    private var context: ModelContext
    private var isRunning = false

    init(modelContext: ModelContext) {
        context = modelContext
        module = loadModel()
        labels = loadLabels()
    }
//    override init() {
//        super.init()
//        module = loadModel()
//        labels = loadLabels()
//    }

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
        let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime)*1000
        let outputs = outputBuffer.floatArray(size: VisionModelContext.outputTensorSize[1])
        let results = getTopN(scores: outputs, count: resultCount, inferenceTime: inferenceTime)
        completionHandler(results, inferenceTime, nil)
        isRunning = false
    }

    private func getTopN(scores: [Float32], count: Int, inferenceTime: Double) -> [InferenceResult] {
        let zippedResults = zip(labels.indices, scores)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(count)
        return sortedResults.map { InferenceResult(score: $0.1, label: labels[$0.0]) }
    }

    private func loadLabels() -> [String] {
        if let filePath = Bundle.main.path(forResource: VisionModelContext.label.name, ofType: VisionModelContext.label.type),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }

    private func loadModel() -> TorchModule? {
        if let filePath = Bundle.main.path(forResource: VisionModelContext.model.name, ofType: VisionModelContext.model.type),
            let module = TorchModule.loadModel(filePath){
            return module
        } else {
            fatalError("Can't find the model with the given path!")
        }
    }
}

extension Predictor {
    enum PredictorError: Swift.Error {
        case invalidModel
        case invalidInputTensor
        case invalidOutputTensor
    }
}
