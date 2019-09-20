import AVFoundation
import UIKit

class ImageClassificationViewController: ViewController {
    lazy var predictor: ImagePredictor = ImagePredictor()
    var cameraController = CameraController()
    @IBOutlet weak var backButton: UIButton!
    @IBOutlet var cameraView: CameraPreviewView!
    var bottomView: BottomView!
//    @IBOutlet var resultView: UITextView!
    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView = BottomView(frame: CGRect(x: 0, y: 100, width: view.frame.width, height: 240))
        view.addSubview(bottomView)
        
//        weak var weakSelf = self
//        cameraController.configPreviewLayer(cameraView)
//        cameraController.videoCaptureCompletionBlock = { buffer, error in
//            if error != nil {
//                // TODO: error handling
//                print(error!)
//                return
//            }
//            weakSelf?.predictor.forward(buffer, completionHandler: { results, _ in
//                DispatchQueue.main.async {
//                    weakSelf?.displayResults(results)
//                }
//            })
//        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
//        cameraController.startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
//        cameraController.stopSession()
    }

    private func displayResults(_ results: [InferenceResult]?) {
        if let results = results {
            var str = ""
            for result in results {
                str += "-[score]: \(result.score), -[label]: \(result.label)\n"
            }
//            resultView.text = str
        }
    }
    @IBAction func onBackClicked(_ sender: Any) {
        navigationController?.popViewController(animated: true)
    }
}
