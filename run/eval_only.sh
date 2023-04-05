name='eval_only'
config='configs/single_baseline_aug1_plus_concat_frms.yaml'

python3 main.py --name ${name} \
--config ${config} \
--eval_only \
--logs-dir logs/eval/${name} \
--resume \

