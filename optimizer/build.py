
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)


def build_vanilla_optimizer(cfg, model, trainloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.LR.BASE_LR)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT*cfg.TRAIN.LR.DELAY , gamma=0.1)
    scheduler = WarmUpLR(lr_scheduler=step_scheduler, warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))
    return optimizer, scheduler


def freeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_bn(model: nn.Module) -> None:
    def set_bn_eval(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)


def unfreeze_bn(model: nn.Module) -> None:
    def set_bn_train(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model.apply(set_bn_train)


class FreezeBackbone(object):
    def __init__(self, model: nn.Module, freeze_epoch=0):
        super().__init__()
        self.model = model
        self.freeze_epoch = freeze_epoch
        self.backbone_name = ['vis_backbone', 'vis_backbone_bk']

    def start_freeze_backbone(self):
        if self.freeze_epoch <= 0:
            return
        for name in self.backbone_name:
            layer = self.model.module._modules[name]
            freeze_params(layer)
            freeze_bn(layer)
            print(f'====> Freeze {name}')

    def on_train_epoch_start(self, epoch) -> None:
        if self.freeze_epoch <= 0:
            return
        if epoch == self.freeze_epoch:
            for name in self.backbone_name:
                layer = self.model.module._modules[name]
                unfreeze_params(layer)
                unfreeze_bn(layer)
                print(f'====> Unfreeze {name}')
