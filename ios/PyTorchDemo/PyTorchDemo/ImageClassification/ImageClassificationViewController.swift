//
//  ImageClassificationViewController.swift
//  PyTorchDemo
//
//  Created by Tao Xu on 9/16/19.
//

import UIKit
import AVFoundation

class ImageClassificationViewController: ViewController {
    lazy var predictor: ImagePredictor = ImagePredictor()
    var cameraController = CameraController()
    @IBOutlet weak var cameraView: CameraPreviewView!
    @IBOutlet weak var resultView: UITextView!
    override func viewDidLoad() {
        super.viewDidLoad()
        weak var weakSelf = self
        cameraController.configPreviewLayer(cameraView)
        cameraController.prepare { error  in
            if let error = error {
                print(error)
                return
            }
            weakSelf?.cameraController.startSession()
        }
        cameraController.videoCaptureCompletionBlock = { buffer, error in
            weakSelf?.predictor.predict( buffer, completionHandler: { results, error in
                DispatchQueue.main.async {
                    weakSelf?.displayResults(results)
                }
            })
        }
    }
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraController.stopSession()
    }
    
    private func displayResults(_ results:[InferenceResult]?) {
        if let results = results {
            var str=""
            for result in results {
                str+="-[score]: \(result.score), -[label]: \(result.label)\n"
            }
            self.resultView.text = str
        }
    }
}
