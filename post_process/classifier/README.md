# Classifier Module

This module do the following tasks:
- Train the Vehicle Color, Type, Direction Classifier.
- Predict labels on visual test data for refinement process.

## Module organization 
- `results`: stores best model weight and prediction results.

## Prepare
Install dependencies
```
cd 'EfficientNet-PyTorch'
pip install -e . 
cd '../'
```


## Train Classifier
```
python train.py
```
Best model weights will be saved in `./results`
Our trained model weights and other result files can downloaded in (https://pan.baidu.com/s/1MyKsQc5RsVmgj0EQx4nE6w?pwd=kc8d code: kc8d)

## Label Prediction
In our model, we predict the one-hot color, type and direction label of target visual tracks, besides, we also predict the one-hot color and type label of the front and back vehicle for target visual tracks. All of this prediction will be used in the post-process.
```
python label_prediction.py                  # predict the color and type label of visual tracks
python direction_label_prediction.py        # predict the direction label of visual tracks
python relation_label_prediction.py         # predict the color and type label of front and back vehicle for the visual tracks
```
The label outputs will be saved in `./results`

