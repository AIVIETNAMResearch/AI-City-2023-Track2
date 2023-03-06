from .utils  import AverageMeter, mean_reciprocal_rank, get_model, get_optimizer, get_scheduler, get_evaluation_steps, sim_matrix
import time
from tqdm import tqdm
import torch
import gc
from dataloader.datasets import get_train_dataloader, get_valid_dataloader
from loss.infonce import InfoNCE
from loss.dcl import DCL
import logging
logging.basicConfig(filename='./log/training.log', filemode='w')
from datetime import datetime
from loss.crossentropy import CrossEntropyLabelSmooth

def valid_fn(valid_dataloader, model, criterion, epoch, device, config, ce_criterion):
    valid_losses = AverageMeter()
    model.eval()
    predictions = []
    start = time.time()
    
    all_image_out = []
    all_text_out = []
    
    
    for step, inputs in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):

        for k, v in inputs.items():
            if not isinstance(v, dict):
                inputs[k] = v.to(device)
                continue
            for k_, v_ in v.items():
                if(isinstance(v_, list)):
                    inputs[k][k_] = [val.to(device) for val in v_]
                elif(isinstance(v_, dict)):
                    for k__, v__ in v_.items():
                        inputs[k][k_][k__] = v__.to(device)
                else:
                    inputs[k][k_] = v_.to(device)

        batch_size = config.general_config.valid_batch_size

        with torch.no_grad():
            image_out, text_out, color_logits, type_logits, motion_logits = model(inputs['video'], inputs['text'], inputs['motion'], inputs['motion_line'])
            #print(image_out.shape, text_out.shape)
            loss_image = criterion(image_out, text_out)
            loss_text = criterion(text_out, image_out)
            loss_color = ce_criterion(color_logits, inputs['color_label'])
            loss_type = ce_criterion(type_logits, inputs['type_label'])
            loss_motion = ce_criterion(motion_logits, inputs['motion_label'])

            loss = loss_image + loss_text + loss_color + loss_type + loss_motion   

        all_image_out.append(image_out)
        all_text_out.append(text_out)
        
        valid_losses.update(loss.item(), batch_size)


        if step % config.general_config.valid_print_frequency == 0 or step == (len(valid_dataloader) - 1):
            print(f'EVAL: [{epoch+1}][{step+1}/{len(valid_dataloader)}] '
                  f'Loss: {valid_losses.val:.4f}({valid_losses.avg:.4f})')
            

    all_image_out = torch.concat(all_image_out)
    all_text_out = torch.concat(all_text_out)

    sim = sim_matrix(all_text_out, all_image_out)
    mrr_score = mean_reciprocal_rank(sim)
    
    return valid_losses, mrr_score

def train_loop(train_folds, valid_folds, device, fold, model_checkpoint_path = None, config=None):
    train_dataloader = get_train_dataloader(config, train_folds)
    valid_dataloader = get_valid_dataloader(config, valid_folds)

    model = get_model(config, config['general_config']['load_checkpoint'])
    model.to(device)

    optimizer = get_optimizer(model, config)

    train_steps_per_epoch = int(len(train_folds) / config.general_config.train_batch_size)
    num_train_steps = train_steps_per_epoch * config.general_config.epochs

    eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      config.general_config.evaluate_n_times_per_epoch)

    scheduler = get_scheduler(optimizer, config, num_train_steps)      
    
    if config.general_config.loss == "InfoNCE":
        criterion = InfoNCE()
    elif config.general_config.loss == "DCL":
        criterion = DCL(temperature=0.5)

    ce_criterion = CrossEntropyLabelSmooth()

    best_score = 0

    for epoch in range(config.general_config.epochs):

        model.train()

        scaler = torch.cuda.amp.GradScaler()

        train_losses = AverageMeter()
        valid_losses = None
        score = None

        start_time = time.time()
        global_step = 0

        for step, inputs in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            for k, v in inputs.items():
                if not isinstance(v, dict):
                    inputs[k] = v.to(device)
                    continue
                for k_, v_ in v.items():
                    if(isinstance(v_, list)):
                        inputs[k][k_] = [val.to(device) for val in v_]
                    elif(isinstance(v_, dict)):
                        for k__, v__ in v_.items():
                            inputs[k][k_][k__] = v__.to(device)
                    else:
                        inputs[k][k_] = v_.to(device)
            # print(inputs)
            batch_size = config.general_config.train_batch_size

            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_out, text_out, color_logits, type_logits, motion_logits = model(inputs['video'], inputs['text'], inputs['motion'], inputs['motion_line'])
                #print(image_out.shape, text_out.shape)
                loss_image = criterion(image_out, text_out)
                loss_text = criterion(text_out, image_out)
                loss_color = ce_criterion(color_logits, inputs['color_label'])
                loss_type = ce_criterion(type_logits, inputs['type_label'])
                loss_motion = ce_criterion(motion_logits, inputs['motion_label'])
                
                loss = loss_image + loss_text + loss_color + loss_type + loss_motion


            if config.general_config.gradient_accumulation_steps > 1:
                loss = loss / config.general_config.gradient_accumulation_steps

            train_losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            
            
            if config.general_config.unscale:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.general_config.max_grad_norm)

            if (step + 1) % config.general_config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if config.scheduler.batch_scheduler:
                    scheduler.step()

            if (step % config.general_config.train_print_frequency == 0) or \
                        (step == (len(train_dataloader) - 1)) or \
                        (step + 1 in eval_steps) or \
                        (step - 1 in eval_steps):

                print(f'Epoch: [{epoch+1}][{step+1}/{len(train_dataloader)}] '
                            f'Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) '
                            f'Grad: {grad_norm:.4f}  '
                            f'LR: {scheduler.get_lr()[0]:.8f}  ')
            
            if (step + 1) in eval_steps:
                valid_losses, mrr_score = valid_fn(valid_dataloader, model, criterion, epoch, device=device, config=config, ce_criterion=ce_criterion)
                model.train()

                print(f'Epoch {epoch+1} - Loss: {valid_losses.val:.4f} - MRR: {mrr_score:.4f}')
                if mrr_score > best_score:
                    best_score = mrr_score

                    torch.save({'model': model.state_dict()}, f"{model_checkpoint_path}/model_ckpt-fold{fold}.pth")
                    print(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')

                unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
                learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

            
        elapsed = time.time() - start_time

        print(f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.4f} '
                    f'avg_val_loss: {valid_losses.avg:.4f} time: {elapsed:.0f}s '
                    f'Epoch {epoch + 1} - Score: {mrr_score:.4f} \n'
                    '=============================================================================\n')

    torch.cuda.empty_cache()
    gc.collect()