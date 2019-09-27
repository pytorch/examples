import Foundation
import UIKit

extension UIViewController {
    func showAlert(_ error: Swift.Error?) {
        let alert = UIAlertController(title: error?.localizedDescription ?? "unknown error", message: nil, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        present(alert, animated: true)
    }
}
