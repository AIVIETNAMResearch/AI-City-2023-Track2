name='single_baseline_aug1_plus_concat_frms_6'
config='configs/single_baseline_aug1_plus_concat_frms.yaml'

epo='400'
freeze='0'
delay='80'
warm='40'
lr='0.01'
num_frames='6'

python3 main.py --name ${name} \
--config ${config} \
--logs-dir logs/${name}_lr${lr}_freeze${freeze}_warm${warm}_decay-${delay} \
TRAIN.LR.BASE_LR ${lr} \
TRAIN.FREEZE_EPOCH ${freeze} \
TRAIN.LR.DELAY ${delay} \
TRAIN.LR.WARMUP_EPOCH ${warm} \
TRAIN.EPOCH ${epo} \
DATA.NUM_FRAMES ${num_frames}