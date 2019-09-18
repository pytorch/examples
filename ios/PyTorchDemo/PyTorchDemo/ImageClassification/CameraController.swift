import Foundation
import AVFoundation

class CameraController: NSObject {
    let inputWidth  = 224
    let inputHeight = 224
    var videoCaptureCompletionBlock:(( [Float32]?, Error?)->Void)?

    func prepare(_ completionHandler: @escaping (Error?)->Void) {
        captureSession.sessionPreset = .high
        sessionQueue.async {
            do {
                self.captureSession.beginConfiguration()
                try self.configCameraInput()
                try self.configCameraOutput()
                self.captureSession.commitConfiguration()
            }catch {
                DispatchQueue.main.async {
                    completionHandler(error)
                }
            }
            DispatchQueue.main.async {
                completionHandler(nil)
            }
        }
    }
    func configPreviewLayer(_ previewView: CameraPreviewView){
        previewView.previewLayer.session      = captureSession
        previewView.previewLayer.connection?.videoOrientation = .portrait
        previewView.previewLayer.videoGravity = .resizeAspectFill
    }
    func startSession(){
        if captureSession.isRunning { return }
        if AVCaptureDevice.authorizationStatus(for: .video) != .authorized {
            if let callback = videoCaptureCompletionBlock {
                callback(nil, CameraControllerError.cameraAccessDenied)
                return
            }
        }
        captureSession.startRunning()
    }
    func stopSession() {
        captureSession.stopRunning()
    }
    private func configCameraInput() throws {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraControllerError.noCamerasAvailable
        }
        let input = try AVCaptureDeviceInput(device: camera)
        if captureSession.canAddInput(input) {
            captureSession.addInput(input);
        } else {
            throw CameraControllerError.invalidInput
        }
    }
    private func configCameraOutput() throws {
        videoOutput.setSampleBufferDelegate(self, queue: bufferQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey):kCMPixelFormat_32BGRA ]
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            throw CameraControllerError.invalidOutput
        }
    }
    private var captureSession  = AVCaptureSession()
    private var videoOutput     = AVCaptureVideoDataOutput()
    private var sessionQueue    = DispatchQueue(label: "session")
    private var bufferQueue     = DispatchQueue(label: "buffer")
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = .portrait
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        guard let normalizedBuffer = pixelBuffer.normalized(inputWidth, inputHeight) else {
            return
        }
        if let callback = videoCaptureCompletionBlock {
            callback(normalizedBuffer,nil)
        }
    }
}

extension CameraController {
    enum CameraControllerError: Swift.Error {
        case cameraAccessDenied
        case noCamerasAvailable
        case invalidInput
        case invalidOutput
        case unknown
    }
}
