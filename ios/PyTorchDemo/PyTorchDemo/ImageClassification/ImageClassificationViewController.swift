import AVFoundation
import UIKit

class ImageClassificationViewController: ViewController {
    lazy var predictor: ImagePredictor = ImagePredictor()
    var cameraController = CameraController()
    @IBOutlet var backButton: UIButton!
    @IBOutlet var cameraView: CameraPreviewView!
    @IBOutlet var bottomView: BottomView!

    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView.setup(with: 3)
        weak var weakSelf = self
        cameraController.configPreviewLayer(cameraView)
        cameraController.videoCaptureCompletionBlock = { buffer, error in
            if error != nil {
                // TODO: error handling
                print(error!)
                return
            }
            weakSelf?.predictor.forward(buffer, completionHandler: { results, _ in
                DispatchQueue.main.async {
                    if let results = results {
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

    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
