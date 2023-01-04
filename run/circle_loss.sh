name='circle_loss'
config='configs/circle_loss.yaml'
cls3='CircleSoftmax'
cls2='CircleSoftmax'
cls1='CircleSoftmax'
clss='64'
clsm='0.3'
metric='CircleLoss'
mets='48'
metm='0.3'
clipm='0.'

python3 main.py --name ${name} \
--config ${config} \
--logs-dir logs/${name}-${metric}_${mets}_${metm} \
MODEL.car_idloss False \
MODEL.mo_idloss False \
MODEL.share_idloss False \
MODEL.HEAD.CAR_CLS ${cls1} \
MODEL.HEAD.MO_CLS ${cls2} \
MODEL.HEAD.SHARED_CLS ${cls3} \
MODEL.HEAD.CLS_LOSS_SCALE ${clss} \
MODEL.HEAD.CLS_LOSS_MARGIN ${clsm} \
MODEL.HEAD.CLIP_LOSS_MARGIN ${clipm} \
MODEL.METRIC.LOSS ${metric} \
MODEL.METRIC.LOSS_SCALE ${mets} \
MODEL.METRIC.LOSS_MARGIN ${metm} 