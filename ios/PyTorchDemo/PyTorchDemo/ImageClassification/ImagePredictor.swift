import UIKit

struct InferenceResult {
    let score: Float32
    let label: String
}

enum ModelContext {
    static let model            = (name: "ResNet18", type: "pt")
    static let label            = (name: "synset_words", type: "txt")
    static let inputTensorSize  = [1,3,224,224]
}

class ImagePredictor: NSObject {
    private var module: TorchModule?
    private var labels: [String] = []
    private var isRunning = false

    override init() {
        super.init()
        module = loadModel()
        labels = loadLabels()
    }

    func forward(_ buffer: [Float32]?, completionHandler: ([InferenceResult]?, Error?) -> Void) {
        guard var tensorBuffer = buffer else {
            return
        }
        if isRunning {
            return
        }
        isRunning = true
        guard let inputTensor = TorchTensor.new(withData: UnsafeMutableRawPointer(&tensorBuffer),
                                                size: ModelContext.inputTensorSize as [NSNumber],
                                                type: .float) else {
            completionHandler([], ImagePredictorError.invalidInputTensor)
            return
        }
        let inputValue = TorchIValue.new(with: inputTensor)
        guard let outputs = module?.forward([inputValue])?.toTensor().floatArray() else {
            completionHandler([], ImagePredictorError.invalidOutputTensor)
            return
        }
        let results = getTopN(scores: outputs, count: 5)
        completionHandler(results, nil)
        isRunning = false
    }

    private func getTopN(scores: [Float32], count: Int) -> [InferenceResult] {
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

    private func loadModel() -> TorchModule? {
        if let filePath = Bundle.main.path(forResource: ModelContext.model.name, ofType: ModelContext.model.type),
            let module = TorchModule.loadTorchscriptModel(filePath) {
            return module
        } else {
            fatalError("Model file was not found.")
        }
    }
}

extension ImagePredictor {
    enum ImagePredictorError: Swift.Error {
        case invalidInputTensor
        case invalidOutputTensor
    }
}
