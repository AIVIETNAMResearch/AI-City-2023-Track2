from torch.optim import AdamW
import torch
from models.baseline_model import VideoTextFeatureExtractor
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup
from torchmetrics import RetrievalMRR
from utils.custom_lr_scheduler import LinearWarmupCosineAnnealingLR

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_model(config, model_checkpoint_path=None):
    model = VideoTextFeatureExtractor(config['arch']['base_settings'], config['arch']['text_head_setting'], config['arch']['vision_head_setting'])

    if model_checkpoint_path is not None:
        state = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(state['model'])
        print(f"Load Check Point: {model_checkpoint_path} successfully!")

    if config.general_config.gradient_checkpointing:
        if model.base.text_model.supports_gradient_checkpointing:
            model.base.text_model.gradient_checkpointing_enable()
        else:
            print(f'{config.arch.base_settings.text_params.model} does not support gradinet checkpointing')
    return model


def get_optimizer_params(model, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    return optimizer_parameters

def get_optimizer(model, config):

    optimizer_parameters = get_optimizer_params(model, weight_decay=config.optimizer.weight_decay)

    optimizer = AdamW(
        optimizer_parameters,
        lr=config.optimizer.learning_rate,
        eps=config.optimizer.eps,
        betas=config.optimizer.betas
    )

    return optimizer

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_evaluation_steps(num_train_steps, n_evaluations):
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps

def get_scheduler(optimizer, config, num_train_steps):

    if config.scheduler.scheduler_type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.constant_schedule_with_warmup.n_warmup_steps
        )
    elif config.scheduler.scheduler_type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.linear_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif config.scheduler.scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.cosine_schedule_with_warmup.n_warmup_steps,
            num_cycles=config.scheduler.cosine_schedule_with_warmup.n_cycles,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler.scheduler_type == 'polynomial_decay_schedule_with_warmup':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.polynomial_decay_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps,
            power=config.scheduler.polynomial_decay_schedule_with_warmup.power,
            lr_end=config.scheduler.polynomial_decay_schedule_with_warmup.min_lr
        )
    elif config.scheduler.scheduler_type == "linear_warmup_cosine_annealing_lr":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
                                                  warmup_epochs=config.scheduler.linear_warmup_cosine_annealing_lr.warmup_epochs,
                                                  max_epochs=config.scheduler.linear_warmup_cosine_annealing_lr.max_epochs)
    else:
        raise ValueError(f'Unknown scheduler: {config.scheduler.scheduler_type}')

    return scheduler

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def mean_reciprocal_rank(sim_mat):
    mrr = RetrievalMRR()
    return mrr(
        sim_mat.flatten(),
        torch.eye(len(sim_mat), device=sim_mat.device).long().bool().flatten(),
        torch.arange(len(sim_mat), device=sim_mat.device)[:, None].expand(len(sim_mat), len(sim_mat)).flatten(),
    )
    pass