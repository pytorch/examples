import UIKit

class NLPViewController: UIViewController {
    @IBOutlet var textView: UITextView!
    @IBOutlet var resultView: ResultView!
    lazy var predictor = NLPPredictor()
    override func viewDidLoad() {
        super.viewDidLoad()
        textView.delegate = self
        resultView.showResult(count: 3, textColor: UIColor(rgb: 0x812CE5))
        resultView.layer.cornerRadius = 12.0
        resultView.layer.masksToBounds = true

        // Do any additional setup after loading the view.
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
            DispatchQueue.global().async {
                self.predictor.forward(content, resultCount: 3, completionHandler: { results, _, error in
                    if error != nil {
                        //
                        return
                    }
                    DispatchQueue.main.async {
                        if let results = results {
                            self.resultView.isHidden = false
                            self.resultView.update(results: results)
                        }
                    }
                })
            }
            return false
        }
        return true
    }
}
