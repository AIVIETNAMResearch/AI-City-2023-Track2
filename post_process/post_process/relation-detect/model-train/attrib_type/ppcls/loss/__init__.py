import copy

import paddle
import paddle.nn as nn
import pdb
from ppcls.utils import logger

from .celoss import CELoss, MixCELoss
from .googlenetloss import GoogLeNetLoss
from .centerloss import CenterLoss
from .emlloss import EmlLoss
from .msmloss import MSMLoss
from .npairsloss import NpairsLoss
from .trihardloss import TriHardLoss
from .triplet import TripletLoss, TripletLossV2
from .supconloss import SupConLoss
from .pairwisecosface import PairwiseCosface
from .dmlloss import DMLLoss
from .distanceloss import DistanceLoss

from .distillationloss import DistillationCELoss
from .distillationloss import DistillationGTCELoss
from .distillationloss import DistillationDMLLoss
from .multilabelloss import MultiLabelLoss
import numpy as np

class CombinedLoss(nn.Layer):
    def __init__(self, config_list):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        self.loss_name_list = []
        
        assert isinstance(config_list, list), (
            'operator config should be a list')
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))
            #self.loss_name_list.append(name)


    def __call__(self, input, batch):
        loss_dict = {}
        loss_dict_final = {}
        flag_sequence = isinstance(input, list) or isinstance(input, tuple)
        if not flag_sequence:
            # only has car head
            input = [input]
            # else has car, year, color, brand head
        label = batch[0]
        #pdb.set_trace()
        for idx,loss_func in enumerate(self.loss_func):
            for attr,weight in enumerate(self.loss_weight[idx]):
#                 weight = weight/len(self.loss_weight[idx])
#                 print("label shape: ",np.squeeze(np.array(label[:,attr]) == 1).shape)
                weight_cls = paddle.ones(label[:,attr].shape,dtype="float32")
            
                if attr == 14:
                    weight_cls = weight_cls + paddle.to_tensor((label[:,attr] == 1),dtype='float32') * 10.0
                    
                weight_cls = paddle.to_tensor(weight_cls,stop_gradient=True)
                loss = loss_func(input[attr], label[:,attr], weight_cls)
#                 loss = {key+str(attr): loss[key] * weight for key in loss}
                loss = {key+str(attr): loss[key] for key in loss}
                loss_dict.update(loss)
#             loss = {key+str(idx): loss[key] for key in loss}
            
        #pdb.set_trace()
        loss_dict_final["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict_final


def build_loss(config):
    module_class = CombinedLoss(copy.deepcopy(config))
    logger.debug("build loss {} success.".format(module_class))
    return module_class
