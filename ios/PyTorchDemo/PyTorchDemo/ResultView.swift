import UIKit

class ItemView: UIView {
    var container: UIStackView!
    var scoreLabel: UILabel!
    var tagLabel: UILabel!
    var progressBar: UIView!

    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override init(frame: CGRect) {
        super.init(frame: frame)

        container = UIStackView()
        container.axis = .vertical
        container.distribution = .fill
        container.alignment = .fill
        container.spacing = 5.0

        let textContainer = UIStackView()
        textContainer.distribution = .fill
        textContainer.alignment = .fill
        textContainer.spacing = 10.0

        tagLabel = UILabel()
        tagLabel.textColor = .white
        scoreLabel = UILabel()
        scoreLabel.textColor = .white
        // flexbox properties
        tagLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        scoreLabel.setContentHuggingPriority(.defaultHigh, for: .horizontal)
        tagLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        scoreLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)

        textContainer.addArrangedSubview(tagLabel)
        textContainer.addArrangedSubview(scoreLabel)

        // progress bar
        progressBar = UIView()
        progressBar.layer.cornerRadius = 6.0
        progressBar.layer.masksToBounds = true
        let gradient = CAGradientLayer()
        progressBar.layer.insertSublayer(gradient, at: 0)

        container.addArrangedSubview(textContainer)
        container.addArrangedSubview(progressBar)

        addSubview(container)
    }

    func update(data: InferenceResult, isTopResult: Bool, progressBarColor: UIColor, textColor: UIColor?) {
        tagLabel.text = data.label
        scoreLabel.text = String(format: "%.2f", data.score)
        scoreLabel.textColor = textColor ?? .white
        tagLabel.textColor = textColor ?? .white
        if isTopResult {
            scoreLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
            tagLabel.font = UIFont.boldSystemFont(ofSize: 18.0)
        } else {
            scoreLabel.font = UIFont.systemFont(ofSize: 14.0)
            tagLabel.font = UIFont.systemFont(ofSize: 14.0)
        }
        container.frame = bounds.insetBy(dx: 20, dy: 5)
        let gradientLayer = progressBar.layer.sublayers?[0] as? CAGradientLayer
        if let gradientLayer = gradientLayer {
            gradientLayer.colors = [progressBarColor.cgColor, UIColor.black.cgColor]
            gradientLayer.startPoint = CGPoint(x: 0, y: 0)
            gradientLayer.endPoint = CGPoint(x: 1, y: 0)
            DispatchQueue.main.async {
                gradientLayer.frame = self.progressBar.bounds
            }
        }
    }
}

class ResultView: UIView {
    var container: UIStackView!
    let colors = [0xE8492B, 0xC52E8B, 0x7C2BDE]
    let spacing: Float = 12.0
    let margin: Float = 20.0
    var topItemViewHeight: Float = 0.0
    var itemViewHeight: Float = 0.0
    var itemViewWidth: Float = 0.0
    var textColor: UIColor?
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }

    func showResult(count: Int, textColor: UIColor) {
        self.textColor = textColor
        itemViewWidth = (Float(frame.width) - Float(margin * 2))
        let tmpHeight = (Float(frame.height) - Float((count + 1) * Int(spacing))) / Float(count)
        topItemViewHeight = tmpHeight * 1.2
        itemViewHeight = tmpHeight - (topItemViewHeight - tmpHeight) / Float(count)
        for _ in 0 ..< count {
            addSubview(ItemView(frame: .zero))
        }
    }

    func update(results: [InferenceResult]) {
        var yOffset: Float = 0
        var height: Float = 0.0
        for index in 0 ..< results.count {
            let resultView = subviews[index] as! ItemView
            // make the first one a little bit bigger
            if index == 0 {
                height = topItemViewHeight
            } else {
                height = itemViewHeight
            }
            resultView.frame = CGRect(x: CGFloat(margin), y: CGFloat(yOffset), width: CGFloat(itemViewWidth), height: CGFloat(height))
            resultView.update(data: results[index], isTopResult: index == 0, progressBarColor: UIColor(rgb: colors[index]), textColor: textColor)
            yOffset += (height + spacing)
        }
    }
}
