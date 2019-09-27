import UIKit

class ImageClassificationItemView: UIView {
    @IBOutlet var contentView: UIView!
    @IBOutlet var resultLabel: UILabel!
    @IBOutlet var scoreLabel: UILabel!
    @IBOutlet var progressBar: UIView!

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    func commonInit() {
        Bundle.main.loadNibNamed("ImageClassificationItemView", owner: self, options: nil)
        contentView.setup(self)
        let gradient = CAGradientLayer()
        progressBar.layer.insertSublayer(gradient, at: 0)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let gradientLayer = progressBar.layer.sublayers?[0] as? CAGradientLayer
        if let gradientLayer = gradientLayer {
            gradientLayer.colors = [progressBar.backgroundColor!.cgColor, UIColor.black.cgColor]
            gradientLayer.startPoint = CGPoint(x: 0, y: 0)
            gradientLayer.endPoint = CGPoint(x: 1, y: 0)
            gradientLayer.frame = progressBar.bounds
        }
    }
}
