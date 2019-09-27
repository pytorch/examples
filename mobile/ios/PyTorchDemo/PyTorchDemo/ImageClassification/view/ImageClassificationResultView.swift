import UIKit

class ImageClassificationResultView: UIView {
    @IBOutlet var contentView: UIView!
    @IBOutlet var containerView: UIStackView!
    var itemViews: [ImageClassificationItemView] = []
    let colors = [0xE8492B, 0xC52E8B, 0x7C2BDE]
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    func commonInit() {
        Bundle.main.loadNibNamed("ImageClassificationResultView", owner: self, options: nil)
        contentView.setup(self)
    }

    func config(_ count: Int) {
        for index in 0 ..< count {
            let itemView = ImageClassificationItemView(frame: .zero)
            if index == 0 {
                itemView.resultLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
                itemView.scoreLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
            } else {
                itemView.resultLabel.font = UIFont.systemFont(ofSize: 14.0)
                itemView.scoreLabel.font = UIFont.systemFont(ofSize: 14.0)
            }
            itemView.progressBar.backgroundColor = UIColor(rgb: colors[index])
            containerView.addArrangedSubview(itemView)
            itemViews.append(itemView)
        }
    }

    func update(results: [InferenceResult]) {
        for index in 0 ..< results.count {
            let itemView = itemViews[index]
            itemView.resultLabel.text = results[index].label
            itemView.scoreLabel.text = String(format: "%.2f", results[index].score)
        }
    }
}
