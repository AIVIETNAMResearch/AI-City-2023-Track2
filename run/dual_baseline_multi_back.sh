name='dual_baseline_multi_back'
config='configs/dual_baseline_multi_back.yaml'

model="dual-text-cat-view-multi-crops"

epo='400'
freeze='0'
delay='80'
warm='40'
lr='0.0003'
#eval='1'

python3 main.py --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024

TRAIN.LR.BASE_LR ${lr} \
TRAIN.FREEZE_EPOCH ${freeze} \
TRAIN.LR.DELAY ${delay} \
TRAIN.LR.WARMUP_EPOCH ${warm} \
TRAIN.EPOCH ${epo} \
#EVAL.EPOCH ${eval} \
