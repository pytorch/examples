//
//  BottomView.swift
//  PyTorchDemo
//
//  Created by Tao Xu on 9/19/19.
//

import UIKit

class ResultView: UIView {
    var container: UIStackView!
    var scoreLabel: UILabel!
    var tagLabel: UILabel!
    var height = 1.0
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    override init(frame: CGRect) {
        super.init(frame: frame)
        container = UIStackView()
        container.axis = .horizontal
        container.distribution = .fill
        container.alignment = .fill
        
        
        tagLabel = UILabel()
        tagLabel.textColor = .white
        tagLabel.textAlignment = .left
        
        scoreLabel = UILabel()
        scoreLabel.textColor = .white
        scoreLabel.textAlignment = .right
        
        container.addArrangedSubview(tagLabel)
        container.addArrangedSubview(scoreLabel)
        addSubview(container)
    }
    func update(score: String, tag: String){
        tagLabel.text = tag
        scoreLabel.text = score
        setNeedsLayout()
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        container.frame = CGRect(x:0, y: 0, width: frame.width, height: frame.height)
    }
    override var intrinsicContentSize: CGSize {
        return CGSize(width: 1.0, height: height)
    }
}

class BottomView: UIView {
    var container: UIStackView!
    var topResultView = ResultView(frame: .zero)
    var middleResultView = ResultView(frame: .zero)
    var bottomResultView = ResultView(frame: .zero)
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        
        container = UIStackView.init(frame: CGRect(x: 0, y: 0, width: frame.width, height: frame.height))
        container.axis = .vertical
        container.distribution = .fill
        container.alignment = .fill
        container.spacing = 10
//        self.layoutMargins = UIEdgeInsets(top: 10, leading: 20, bottom: 20, trailing: 10)
        
        topResultView.backgroundColor = .red
        middleResultView.backgroundColor = .green
        bottomResultView.backgroundColor = .blue
        
        container.addArrangedSubview(topResultView)
        container.addArrangedSubview(middleResultView)
        container.addArrangedSubview(bottomResultView)
        addSubview(container)
        update()
    }
    
    func update() {
        topResultView.frame.size.height = 1
        topResultView.setContentHuggingPriority(UILayoutPriority(100), for: .vertical)
        topResultView.update(score: "12", tag: "asdfasdfasdfasdf")
        middleResultView.frame.size.height = 1
        middleResultView.setContentHuggingPriority(UILayoutPriority(200), for: .vertical)
        middleResultView.update(score: "12", tag: "asdfasdfasdfasdf")
        bottomResultView.frame.size.height = 1
        bottomResultView.setContentHuggingPriority(UILayoutPriority(300), for: .vertical)
        bottomResultView.update(score: "12", tag: "asdfasdfasdfasdf")
        setNeedsLayout()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
