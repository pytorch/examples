import UIKit

class NLPItemView: UIView {
    @IBOutlet var contentView: UIView!
    @IBOutlet var resultLabel: UILabel!
    @IBOutlet var scoreLabel: UILabel!
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    func commonInit() {
        Bundle.main.loadNibNamed("NLPItemView", owner: self, options: nil)
        contentView.setup(self)
    }
}
