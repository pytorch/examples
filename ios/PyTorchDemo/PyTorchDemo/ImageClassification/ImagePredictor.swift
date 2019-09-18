import UIKit

struct InferenceResult {
    let score: Float32
    let label: String
}

enum ModelContext {
    static let model = (name:"model", type:"pt")
    static let label = (name:"synset_words", type:"txt")
}

class ImagePredictor : NSObject {
    private var module: TorchModule?
    private var labels: [String] = []
    private var isRunning = false
    
    override init() {
        super.init()
        module = loadModel()
        labels = loadLabels()
    }
    func predict(_ buffer: [Float32]?, completionHandler:(([InferenceResult]?, Error?)->Void)){
        guard var tensorBuffer = buffer else {
            return
        }
        if isRunning {
            return
        }
        isRunning = true
        guard let inputTensor = TorchTensor.new(with: .float, size: [1,3,224,224], data: UnsafeMutableRawPointer(&tensorBuffer)) else {
            completionHandler([],ImagePredictorError.invalidInputTensor)
            return
        }
        let inputValue = TorchIValue.new(with: inputTensor)
        guard let outputTensor = module?.forward([inputValue])?.toTensor() else {
            completionHandler([],ImagePredictorError.invalidOutputTensor)
            return
        }
        let results = getTopN(results: outputTensor, count: 5);
        completionHandler(results,nil)
        isRunning = false;
    }
    private func getTopN(results: TorchTensor, count: Int) -> [InferenceResult] {
        let resultCount: UInt = results.sizes[1].uintValue
        var scores: [Float32] = []
        for index: UInt in 0..<resultCount {
            if let score = results[0]?[index]?.item()?.floatValue {
                scores.append(score)
            }
        }
        let zippedResults = zip(labels.indices,scores)
        
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(count)
        return sortedResults.map({ result in InferenceResult(score: result.1, label: labels[result.0])})
    }
    
    private func loadLabels() -> [String]{
        if let filePath = Bundle.main.path(forResource: "synset_words", ofType: "txt"),
            let labels = try? String(contentsOfFile:  filePath){
            return labels.components(separatedBy: .newlines);
        }else {
            fatalError("Label file was not found.")
        }
    }
    private func loadModel() -> TorchModule? {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
            let module = TorchModule.loadTorchscriptModel(filePath) {
            return module
        }else {
            fatalError("Model file was not found.")
        }
    }
}

extension ImagePredictor {
    enum ImagePredictorError:Swift.Error {
        case invalidInputTensor
        case invalidOutputTensor
    }
}
