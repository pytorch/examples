import UIKit
import AVFoundation

class CameraPreviewView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer {
        guard let layer = self.layer as? AVCaptureVideoPreviewLayer else {
            fatalError("AVCaptureVideoPreviewLayer is expected")
        }
        return layer;
    }
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self;
    }
}
