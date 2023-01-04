name='eval_only'
config='configs/dual_baseline_aug1.yaml'

python3 main.py --name ${name} \
--config ${config} \
--eval_only \
--logs-dir logs/eval/${name} \
--resume \

