import Foundation

struct InferenceResult {
    let score: Float32
    let label: String
}

enum PredictorError: Swift.Error {
    case invalidModel
    case invalidInputTensor
    case invalidOutputTensor
}

protocol Predictor {
    var module: TorchModule? { get set }
    var labels: [String] { get set }
}

extension Predictor {
    func loadLabels(name: String) -> [String] {
        if let filePath = Bundle.main.path(forResource: name, ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }

    func loadModel(name: String) -> TorchModule? {
        if let filePath = Bundle.main.path(forResource: name, ofType: "pt"),
            let module = TorchModule.loadModel(filePath) {
            return module
        } else {
            fatalError("Can't find the model with the given path!")
        }
    }

    func getTopN(scores: [Float32], count: Int, inferenceTime _: Double) -> [InferenceResult] {
        let zippedResults = zip(labels.indices, scores)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(count)
        return sortedResults.map { InferenceResult(score: $0.1, label: labels[$0.0]) }
    }
}
