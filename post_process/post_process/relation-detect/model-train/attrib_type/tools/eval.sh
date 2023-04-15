#!/usr/bin/env bash

# for single card eval
export CUDA_VISIBLE_DEVICES=0
#python3.7 tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml
#python3.7 tools/eval.py -c ./ppcls/configs/truck_resnet50.yaml


# test inference model
python3.7 tools/eval.py -c ./ppcls/configs/truck_resnet50.yaml
#python3.7 tools/test_inference_model_truck2.py

# for multi-cards eval
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml
