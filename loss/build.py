
from .triplet import TripletLoss
from .crossentropy import CrossEntropyLabelSmooth, cross_entropy
from .softmaxs import Linear, NormSoftmax, CosSoftmax, ArcSoftmax, CircleSoftmax
from .pairwise import CosFacePairLoss, CosFacePairLoss_v2, CirclePairLoss
from pytorch_metric_learning.losses import CircleLoss
from . import softmaxs


def build_softmax_cls(model_cfg, loss_type):
    num_features = model_cfg.EMBED_DIM
    num_class = model_cfg.NUM_CLASS
    cls_scale = model_cfg.HEAD.CLS_LOSS_SCALE
    cls_margin = model_cfg.HEAD.CLS_LOSS_MARGIN
    classifier = getattr(softmaxs, loss_type)(num_features, num_class, cls_scale, cls_margin)
    return classifier
