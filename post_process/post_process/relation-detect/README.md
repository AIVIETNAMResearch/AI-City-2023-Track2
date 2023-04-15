# Related Vehicles Detection
In order to better refine the original similarity matrix, we detect the front, back and near vehicles for each tracked vehicles, and use these vehicles attribute information i.e color and type to check whether the query and track are consistent and adjust the similarity score. This module consist of `model_train` and `relation-det`. Please note that these two parts are both implemented with PaddlePaddle.
```
.
├── AICITY2022
├── model-train
│   ├── attrib_color
│   ├── attrib_type
│   └── det
└── relation-det
    ├── all_track_to_query.txt
    ├── final_mat
    ├── get_relation_attrib
    ├── get_relation_mat
    ├── get_track_relation
    ├── intersection
    ├── test_queries.json
    └── test_tracks.json
```
## preparation
We take the cropped target vehicle's images as the input to train the color and type classifiers, so first you need to prepare the training data and put them into the `AICITY2022` folder. You can download these data files in [aic22_randcrop.tar](https://pan.baidu.com/s/1LwSE_UZXesP29BXLWCvnNg?pwd=8r43 code: 8r43). After that, you need to create the soft links to this folder in the `attrib_type/dataset` and `attrib_color/dataset`.

## model-train
This part consist of train the color, type attribute classifiers with PaddleCls and vehicle detector with PaddleDet.
```
cd attrib_color
sh train.sh  
sh eval.sh
cd attib_type
sh train.sh
sh eval.sh
```
For the vehicle detector, we simply use the PP-YOLOE detection model implemented with PaddlePaddleDetection with the coco pretrained model weight. The config we used here is `ppyoloe_crn_x_300e_coco.yml ` which you can see in https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe. The pretrained model weight we used here is https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams. Please note that there is no need to train or fine-tune the model, and the only thing we need to do is inference the track2 dataset with this well-trained model. To reproduce the result,you only need to clone the PaddlePaddleDetection repo into the det folder, and make sure the data path is correct, then run
```
sh infer_aicity.sh
```

After running this command, you will obtain the vehicle detection results in 'bboox.json', which you can download in [bbox.json](https://pan.baidu.com/s/1bNEb_EPa1OddAdXjeUuNqQ?pwd=vuld code: vuld), and it contains the vehicle detection results of each frame of the video. Finally, put the bbox.json into relation-det.

## relation-det
After obtain the attribute classifier and vehicle detector, we first generate the mask of target vehicle's lane, and take the IoU as the measure to check whether a vehicle is the front, back or near vehicle of target vehicle. And then, we predict these car's color and type labels with the classifier trained above. In order to increase the robustness of the prediction results, we also ensemble the color and type label prediction with the inference results of the color and type classifier trained with query labels before. You can run the following command:
```
cd get_track_relation
sh run.sh
cd get_relation_attrib
sh run.sh
cd get_relation_mat
sh run.sh
cd intersection
sh run.sh
```
After that, you can obtain the final confusion matrixes of the target related vehicles in the `final_mat` folder, which will be used in the following post-process.