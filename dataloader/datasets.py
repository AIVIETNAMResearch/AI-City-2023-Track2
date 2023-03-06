import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import get_motion_img
import os
import torchvision.transforms as transforms

class Track2CustomDataset(Dataset):
    def __init__(self, video_params, data_tracks, tokenizer, max_len, transforms, config, mode="train"):
        
        self.samples = data_tracks
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.video_params = video_params
        self.mode = mode
        self.config = config
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]

        if self.mode == "train":

            final, motion, motion_line = self.image_features(sample)
            text_inputs = self.lang_features(sample)

            sample = {
                'text': text_inputs,
                'video': final,
                'motion': motion,
                'motion_line': motion_line,
                'color_label': sample['colors'],
                'type_label': sample['type'],
                'motion_label': sample['motion']

            }

            return sample
        
        if self.mode == "infer_text":

            text_inputs = self.lang_features(sample)
            return {'text': text_inputs}
        
        if self.mode == "infer_video":
            final, motion, motion_line = self.image_features(sample)
            return {
                'video': final,
                'motion': motion,
                'motion_line': motion_line
            }    
    
    def image_features(self, sample):
        frames_path, boxes = sample['frames'], sample['boxes']
            
        veh_imgs, motion_line, motion = get_motion_img(self.config['general_config']['data_dir'], frames_path, boxes, self.config['arch']['base_settings']['video_params']['num_frames'])

        if self.transforms:
            veh_imgs = [self.transforms(img.astype(np.float32)) for img in veh_imgs]
            motion_line = self.transforms(motion_line.astype(np.float32))
            motion = self.transforms(motion.astype(np.float32))
    
        veh_imgs = torch.stack(veh_imgs)
        
        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'], self.video_params['input_res']])
        final[: veh_imgs.shape[0]] = veh_imgs

        return final, motion, motion_line

    def lang_features(self, sample):
        nl_descriptions = sample['nl']        
        text_inputs = []
        for idx, text in enumerate(nl_descriptions):
            # print("text: ", text, ", idx: ", idx)
            tokenized_inp = self.tokenizer.encode_plus(
                                text,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
            text_inputs.append({
                'input_ids': torch.LongTensor(tokenized_inp['input_ids']),
                'attention_mask': torch.LongTensor(tokenized_inp['input_ids'])
            })        
        
        return text_inputs

def videotext_collate_fn(batch_data):
    frames = torch.stack([item['video'] for item in batch_data])
    motion = torch.stack([item['motion'] for item in batch_data])
    motion_line = torch.stack([item['motion_line'] for item in batch_data])
    input_ids = torch.stack([cap['input_ids'] for item in batch_data for cap in item['text']])
    attention_mask = torch.stack([cap['attention_mask'] for item in batch_data for cap in item['text']])
    color_label = torch.LongTensor([item['color_label'] for item in batch_data])
    type_label = torch.LongTensor([item['type_label'] for item in batch_data])
    motion_label = torch.LongTensor([item['motion_label'] for item in batch_data])

    return {'video': frames, 'text': {'input_ids': input_ids, 'attention_mask': attention_mask}, 'motion': motion, 'motion_line': motion_line,
            'color_label': color_label, 'type_label': type_label, 'motion_label': motion_label}

def text_collate_fn(batch_data):
    input_ids = torch.stack([cap['input_ids'] for item in batch_data for cap in item['text']])
    attention_mask = torch.stack([cap['attention_mask'] for item in batch_data for cap in item['text']])
    
    return {'text': {'input_ids': input_ids, 'attention_mask': attention_mask}}

def video_collate_fn(batch_data):
    frames = torch.stack([item['video'] for item in batch_data])
    motion = torch.stack([item['motion'] for item in batch_data])
    motion_line = torch.stack([item['motion_line'] for item in batch_data])
    return {'video': frames, 'motion': motion, 'motion_line': motion_line}



def get_transforms(img_size, train, size=1):
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(img_size * size, scale=(0.8, 1)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size * size, img_size * size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


def get_train_dataloader(config, df):
    dataset = Track2CustomDataset(data_tracks=df, 
                                  video_params=config.arch.base_settings.video_params,
                                  tokenizer=config.general_config.tokenizer,
                                  max_len=int(config.general_config.max_len),
                                  transforms=get_transforms(config.arch.base_settings.video_params.input_res, train=True),
                                  config=config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.general_config.train_batch_size,
        num_workers=config.general_config.n_workers,
        collate_fn=videotext_collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    return dataloader
    
def get_valid_dataloader(config, df):
    dataset = Track2CustomDataset(data_tracks=df, 
                                  video_params=config.arch.base_settings.video_params,
                                  tokenizer=config.general_config.tokenizer,
                                  max_len=int(config.general_config.max_len),
                                  transforms=get_transforms(config.arch.base_settings.video_params.input_res, train=False),
                                  config=config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.general_config.valid_batch_size,
        num_workers=config.general_config.n_workers,
        collate_fn=videotext_collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    return dataloader

def get_infer_dataloader(config, df_video, df_text):
    text_dataset = Track2CustomDataset(data_tracks=df_text,
                                       video_params=config.arch.base_settings.video_params,
                                       tokenizer=config.general_config.tokenizer,
                                       max_len=int(config.general_config.max_len),
                                       transforms=get_transforms(config.arch.base_settings.video_params.input_res, train=False),
                                       config=config,
                                       mode="infer_text")
    
    video_dataset = Track2CustomDataset(data_tracks=df_video,
                                        video_params=config.arch.base_settings.video_params,
                                        tokenizer=config.general_config.tokenizer,
                                        max_len=int(config.general_config.max_len),
                                        transforms=get_transforms(config.arch.base_settings.video_params.input_res, train=False),
                                        config=config,
                                        mode="infer_video")
    
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=config.general_config.valid_batch_size,
        num_workers=config.general_config.n_workers,
        collate_fn=text_collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    video_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.general_config.valid_batch_size,
        num_workers=config.general_config.n_workers,
        collate_fn=video_collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    return video_dataloader, text_dataloader