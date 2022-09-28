#!/usr/bin/env bash

python main_profile_mtgpu.py -a resnet50 -b 16 --epochs 90 --mtgpu [imagenet-folder with train and val folders]

