//
//  NLPModelCard.swift
//  PyTorchDemo
//
//  Created by Tao Xu on 9/24/19.
//

import UIKit

class NLPModelCard: UIView {
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    override init(frame: CGRect) {
        super.init(frame: frame)
    }
    static func show() {
        if let window = UIApplication.shared.delegate?.window {
            let backgroundView = VisionModelCard(frame:window!.bounds)
            backgroundView.backgroundColor = .clear
            let nibs = Bundle.main.loadNibNamed("NLPModelCard", owner: backgroundView, options: nil)
            if let card = nibs?.first as? UIView {
                card.center = backgroundView.center
                card.layer.cornerRadius = 12.0
                card.layer.masksToBounds = true
                backgroundView.addSubview(card)
            }
            window?.addSubview(backgroundView)
        }
    }
    @IBAction func onCancelClicked(_ sender: Any) {
        removeFromSuperview()
    }
}
