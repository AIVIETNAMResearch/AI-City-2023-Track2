# Track2-Vehicle_Retrieval

The directory structures in data is as follows

```
data
└── AIC23_Track2_NL_Retrieval
    ├── baseline
    └── data
        ├── clip_feats
        ├── data
        │   ├── bk_map
        │   ├── motion_heatmap
        │   ├── motion_line
        │   ├── motion_map
        │   └── motion_map_iou
        ├── tracks
        ├── train
        ├── validation
        ├── train-tracks_nlpaug.json    
        ├── train-tracks_nlpaug_2.json  
        ├── train-tracks_nlpaug_3.json
        └── ...

```

## Train

The configuration files are in `configs` and train different models by (set up the right data path first):

```
bash run/single_baseline_aug1.sh
bash run/single_baseline_aug1_plus.sh
bash run/single_baseline_aug2.sh
bash run/circle_loss.sh
bash run/view_triplet_hard.sh
bash run/dual_baseline_aug3.sh
```

You can also change the `RESTORE_FROM` in your configuration file to checkpoints, and load checkpoints to eval (download the checkpoints first).
Take `dual_baseline_aug1` as an example:

```
bash run/eval_only.sh
```