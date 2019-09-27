import UIKit

class NLPViewController: UIViewController {
    @IBOutlet var textView: UITextView!
    @IBOutlet var resultView: NLPResultView!
    lazy var predictor = NLPPredictor()
    override func viewDidLoad() {
        super.viewDidLoad()
        textView.delegate = self
        resultView.config(3)
    }

    @IBAction func onInfoClicked(_: Any) {
        textView.resignFirstResponder()
        NLPModelCard.show()
    }

    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}

extension NLPViewController: UITextViewDelegate {
    func textView(_ textView: UITextView, shouldChangeTextIn _: NSRange, replacementText text: String) -> Bool {
        if text == "\n" {
            resultView.isHidden = true
            textView.resignFirstResponder()
            let content = textView.text!
            weak var weakSelf = self
            DispatchQueue.global().async {
                self.predictor.forward(content, resultCount: 3, completionHandler: { results, _, error in
                    if error != nil {
                        weakSelf?.showAlert(error)
                        return
                    }
                    DispatchQueue.main.async {
                        if let results = results {
                            weakSelf?.resultView.isHidden = false
                            weakSelf?.resultView.update(results: results)
                        }
                    }
                })
            }
            return false
        }
        return true
    }
}
