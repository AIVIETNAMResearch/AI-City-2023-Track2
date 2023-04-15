CONFIG = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 0,
    'image_size': (224,224),
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    'score_thres': 0.5,
    'seed': 88,

    'train': {
        'batch_size': 16,
        'num_epochs': 5,    # set 5 for test
    },
    'val':{
        'batch_size': 8,
    }
}
cfg_veh = CONFIG.copy()
cfg_col = CONFIG.copy()
cfg_dir = CONFIG.copy()


cfg_veh.update({
    'NUM_CLASSES': 6,
    'WEIGHT': './results/veh_classifier.pth'
})

cfg_col.update({
    'NUM_CLASSES': 8,
    'WEIGHT': './results/col_classifier.pth'
})
cfg_dir.update({
    'NUM_CLASSES': 4,
    'WEIGHT': './results/dir_classifier.pth'
})
