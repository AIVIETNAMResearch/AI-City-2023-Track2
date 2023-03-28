name='dual_baseline_aug3_multiframes'
config='configs/dual_baseline_aug3_multiframes.yaml'

model="dual-text-cat-multiframes"

epo='20'
freeze='40'
delay='80'
warm='40'
lr='0.006'
eval='5'

python3 main.py --name ${name} --resume \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024 \
TRAIN.LR.BASE_LR ${lr} \
TRAIN.LR.DELAY ${delay} \
TRAIN.LR.WARMUP_EPOCH ${warm} \
TRAIN.EPOCH ${epo} \
EVAL.EPOCH ${eval} \
