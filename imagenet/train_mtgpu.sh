#!/usr/bin/env bash

python main_mtgpu.py -a resnet50 -b 16 --epochs 90 --mtgpu [imagenet-folder with train and val folders]

#python main_mtgpu.py -a resnet50 --mtgpu --test_perf [imagenet-folder with train and val folders]
