# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle
import pdb

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
import numpy as np

def classification_eval(engine, epoch_id=0):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    metric_key = None
    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0]).astype("float32")
#         if not engine.config["Global"].get("use_multilabel", False):
#             batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        # image input
        out = engine.model(batch[0])
        # calc loss
        if engine.eval_loss_func is not None:
            loss_dict = engine.eval_loss_func(out, batch[1:])
            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0], batch_size)

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        # calc metric
        if engine.eval_metric_func is not None:
#             print(out)
#             print(batch[1])
            
            pred_ori = out
            label_ori = batch[1]
            class_num = engine.config["Arch"]["class_num"]
            for dim in range(len(class_num)):
                
                metric_dict = {}
                pred_new = pred_ori[dim]
                label_new = label_ori[:,dim]
                label_new = paddle.unsqueeze(label_new, axis=1)
                
                if paddle.distributed.get_world_size() > 1:
                    label_list = []
                    paddle.distributed.all_gather(label_list, label_new)
                    labels = paddle.concat(label_list, 0)

                    if isinstance(pred_new, dict):
                        if "logits" in pred_new:
                            pred_new = pred_new["logits"]
                        elif "Student" in pred_new:
                            pred_new = pred_new["Student"]
                        else:
                            msg = "Error: Wrong key in out!"
                            raise Exception(msg)
                    if isinstance(pred_new, list):
                        pred = []
                        for x in pred_new:
                            pred_list = []
                            paddle.distributed.all_gather(pred_list, x)
                            pred_x = paddle.concat(pred_list, 0)
                            pred.append(pred_x)
                    else:
                        pred_list = []
                        paddle.distributed.all_gather(pred_list, pred_new)
                        pred = paddle.concat(pred_list, 0)
#                     if accum_samples > total_samples and not engine.use_dali:
#                         pred = pred[:total_samples + current_samples -
#                                     accum_samples]
#                         labels = labels[:total_samples + current_samples -
#                                         accum_samples]
#                         current_samples = total_samples + current_samples - accum_samples

                    labels = labels.numpy()
                    valid_mask = labels>-1
                    labels = labels[valid_mask[:,0],:]
                    pred = pred.numpy()
                    pred = pred[valid_mask[:,0],:]
                    if labels.shape[0] > 0 :
                        labels = paddle.to_tensor(labels)
                        pred = paddle.to_tensor(pred)
                        metric_dict = engine.eval_metric_func(pred, labels)
                else:
                    metric_dict = engine.eval_metric_func(pred_new, label_new)
                
                
                for key in metric_dict:
                    if metric_key is None:
                        metric_key = key+"_"+str(dim)
                    if key+"_"+str(dim) not in output_info:
                        output_info[key+"_"+str(dim)] = AverageMeter(key, '7.5f')

                    output_info[key+"_"+str(dim)].update(metric_dict[key].numpy()[0],
                                            current_samples)
#                     print("output_info: ",output_info[key+"_"+str(dim)])
                    
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in sorted(output_info)
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg) for key in sorted(output_info)
    ])
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # do not try to save best eval.model
    if engine.eval_metric_func is None:
        return -1
    # return 1st metric in the dict
    return output_info[metric_key].avg
