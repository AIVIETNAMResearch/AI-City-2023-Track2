name='submit'
config='configs/single_baseline_aug1.yaml'

#CUDA_VISIBLE_DEVICES=0,1,2,3
python3 get_submit.py --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
TEST.BATCH_SIZE 128 \
