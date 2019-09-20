import UIKit

class NLPViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }
    

    @IBAction func onBackClicked(_ sender: Any) {
        self.navigationController?.popViewController(animated: true)
    }

}
