import AVFoundation
import UIKit

class ImageClassificationViewController: ViewController {
    @IBOutlet var cameraView: CameraPreviewView!
    @IBOutlet var bottomView: ImageClassificationResultView!
    @IBOutlet var benchmarkLabel: UILabel!
    @IBOutlet var indicator: UIActivityIndicatorView!
    lazy var predictor = ImagePredictor()
    var cameraController = CameraController()

    override func viewDidLoad() {
        super.viewDidLoad()
        bottomView.config(3)
        weak var weakSelf = self
        cameraController.configPreviewLayer(cameraView)
        cameraController.videoCaptureCompletionBlock = { buffer, error in
            if error != nil {
                weakSelf?.showAlert(error)
                return
            }
            weakSelf?.predictor.forward(buffer, resultCount: 3, completionHandler: { results, inferenceTime, error in
                if error != nil {
                    weakSelf?.showAlert(error)
                    return
                }
                DispatchQueue.main.async {
                    weakSelf?.indicator.isHidden = true
                    weakSelf?.bottomView.isHidden = false
                    weakSelf?.benchmarkLabel.isHidden = false
                    if let results = results {
                        weakSelf?.benchmarkLabel.text = String(format: "%.3fms", inferenceTime)
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

    @IBAction func onInfoBtnClicked(_: Any) {
        VisionModelCard.show()
    }

    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
