import UIKit

class ResultView: UIView {
    var container: UIStackView!
    var scoreLabel: UILabel!
    var tagLabel: UILabel!
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        layer.cornerRadius = 8.0
        layer.masksToBounds = true;
        
        container = UIStackView()
        container.axis = .horizontal
        container.distribution = .fill
        container.alignment = .fill
        container.spacing = 20
        
        tagLabel = UILabel()
        tagLabel.textColor = .white
        scoreLabel = UILabel()
        scoreLabel.textColor = .white
        // flexbox properties
        tagLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        scoreLabel.setContentHuggingPriority(.defaultHigh, for: .horizontal)
        tagLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        scoreLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)
    
        container.addArrangedSubview(tagLabel)
        container.addArrangedSubview(scoreLabel)
        addSubview(container)
    }    
    func update(data: InferenceResult, isTopResult: Bool){
        tagLabel.text   = data.label
        scoreLabel.text = String(format: "%.2f", data.score)
        if (isTopResult){
            scoreLabel.font =  UIFont.boldSystemFont(ofSize: 18.0)
            tagLabel.font =  UIFont.boldSystemFont(ofSize: 18.0)
        }else {
            scoreLabel.font =  UIFont.systemFont(ofSize: 14.0)
            tagLabel.font =  UIFont.systemFont(ofSize: 14.0)
        }
        let containerFrame = bounds.insetBy(dx: 20, dy: 5)
        container.frame = containerFrame
    }
}

class BottomView: UIView {
    var container: UIStackView!
    var resultViews:[ResultView] = []
    let colors = [0xe8492b,0xc52e8b,0x7c2bde]
    let spacing: Float = 10.0
    let margin: Float  = 20.0
    var topResultViewHeight: Float = 0.0
    var resultViewHeight: Float = 0.0
    var resultViewWidth: Float  = 0.0
    required init?(coder aDecoder: NSCoder) {
        super.init(coder:aDecoder)
    }
    func setup(with count: Int){
        resultViewWidth  = ( Float(frame.width) - Float(margin*2) )
        let tmpHeight = ( Float(frame.height) - Float(count*Int(spacing)) ) / Float(count)
        topResultViewHeight = tmpHeight*1.2
        resultViewHeight = tmpHeight - (topResultViewHeight - tmpHeight)/Float(count)
        for _ in 0..<count {
            addSubview(ResultView(frame: .zero))
        }
    }
    func update(results: [InferenceResult] ) {
        var yOffset: Float = spacing
        var height: Float = 0.0
        for index in 0..<results.count {
            let resultView = subviews[index] as! ResultView
            // make the first one a little bit bigger
            if index == 0 {
                height = topResultViewHeight
            } else {
                height = resultViewHeight
            }
            resultView.frame = CGRect(x: CGFloat(margin), y: CGFloat(yOffset), width: CGFloat(resultViewWidth), height: CGFloat(height))
            resultView.backgroundColor = UIColor(rgb: colors[index])
            resultView.update(data: results[index], isTopResult: index == 0)
            yOffset += (height + spacing)
        }
    }
}
