export CUDA_VISIBLE_DEVICES=0
#python3 tools/export_model.py \
#    -c ./ppcls/configs/truck_resnet50.yaml \
#    -o Global.pretrained_model=weights/latest

python3 tools/export_model.py \
    -c ./ppcls/configs/car_type_r50.yaml \
    -o Global.pretrained_model=output/ResNet50_vd/best_model
