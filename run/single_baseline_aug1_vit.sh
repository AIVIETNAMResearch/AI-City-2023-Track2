name='single_baseline_aug1_vit'
config='configs/single_baseline_aug1_vit.yaml'

epo='400'
freeze='0'
delay='80'
warm='40'
lr='0.0003'

python3 main.py --name ${name} \
--config ${config} \
--logs-dir logs/${name}_lr${lr}_freeze${freeze}_warm${warm}_decay-${delay} \
TRAIN.LR.BASE_LR ${lr} \
TRAIN.FREEZE_EPOCH ${freeze} \
TRAIN.LR.DELAY ${delay} \
TRAIN.LR.WARMUP_EPOCH ${warm} \
TRAIN.EPOCH ${epo} \
