import AVFoundation
import UIKit

class ImageClassificationViewController: ViewController {
    lazy var predictor = Predictor(modelContext: ModelContext(model: (name:"ResNet", type:"pt"),
                                                              label: (name: "Label", type: "txt"),
                                                              inputTensorSize: [1,3,224,224],
                                                              outputTensorSize: [1,1000]))
    var cameraController = CameraController()
    @IBOutlet var cameraView: CameraPreviewView!
    @IBOutlet var bottomView: ResultView!
    @IBOutlet weak var inferenceTimeLabel: UILabel!
    @IBOutlet weak var indicator: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView.showResult(count: 3, textColor: .white)
        weak var weakSelf = self
        cameraController.configPreviewLayer(cameraView)
        cameraController.videoCaptureCompletionBlock = { buffer, error in
            if error != nil {
                // TODO: error handling
                print(error!)
                return
            }
            weakSelf?.predictor.forward(buffer, resultCount:3, completionHandler: { results, inferenceTime, error in
                if error != nil {
                    // TODO: error handling
                    print(error!)
                    return
                }
                DispatchQueue.main.async {
                    weakSelf?.indicator.isHidden = true
                    if let results = results {
                        weakSelf?.inferenceTimeLabel.text = String(format: "%.3fms", inferenceTime)
                        weakSelf?.bottomView.update(results: results)
                    }
                }
            })
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(true, animated: false)
        cameraController.startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraController.stopSession()
    }

    @IBAction func onInfoBtnClicked(_ sender: Any) {
        VisionModelCard.show()
    }
    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
