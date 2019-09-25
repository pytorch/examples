import UIKit

class NLPViewController: UIViewController {
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var resultView: ResultView!
    override func viewDidLoad() {
        super.viewDidLoad()
        textView.delegate = self
        resultView.showResult(count: 3, textColor: UIColor(rgb: 0x812ce5))
        resultView.layer.cornerRadius = 12.0
        resultView.layer.masksToBounds = true

        // Do any additional setup after loading the view.
        
    }
    @IBAction func onInfoClicked(_ sender: Any) {
        textView.resignFirstResponder()
        NLPModelCard.show()
    }
    @IBAction func onBackClicked(_: Any) {
        navigationController?.popViewController(animated: true)
    }
}
extension NLPViewController: UITextViewDelegate {
    func textView(_ textView: UITextView, shouldChangeTextIn range: NSRange, replacementText text: String) -> Bool {
        if text == "\n" {
            resultView.isHidden = false
            textView.resignFirstResponder()
            return false;
        }
        return true;
    }
    
}



