name='view_triplet_hard'
config='configs/view_triplet_hard.yaml'

view_loss='Triplet'
model='dual-stream-v2-view'
m='0.'
soft='False'
python3 main.py --name ${name} \
--config ${config} \
--logs-dir logs/${name}_${model}_${view_loss}-${m}-soft${soft} \
MODEL.HEAD.NLP_VIEW_LOSS ${view_loss} \
MODEL.NAME ${model} \
MODEL.HEAD.NLP_VIEW_LOSS_MARGIN ${m} \
MODEL.HEAD.NLP_VIEW_SOFT ${soft}