//
//  ViewController.swift
//  PyTorchDemo
//
//  Created by Tao Xu on 9/16/19.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        var predictor = LibTorchPredictor()
        predictor.loadTorchScriptModel("")
        
    }


}

