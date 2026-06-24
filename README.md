This repo is a small test project to fine-tune a small depth astimation model (depth anything).
The code uses lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320 data set to fine tune depth estimation for drone setup.

# Steps

## divide the data
Split the data to train (70%) validation (15%) and test (15%)


## Evaluate pre-trained model
We run the model the test set.
Before computing metrics, predicated and GT depth are alighed to exclude scale and shift from the calculations and standardize everything to a unified measure. following https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide

### Metrics
* absRel - Absolute Relative Error - measuring how much the predicted distances differ from the true ones on average in percentage terms
* rmse - Root Mean Square Error
* deltas - Threashold accuracy. /delta1 measures the percentage of predicted pixels that differ from the true pixels by no more than 25%. 

## Traing
LoRA adaption is used for this fine-tuning, to reduce the computation. Some of the layers will get an adaption. The adation is a set of additinal wieghts for the input. this addiotnal matrix is computed base on 2 relatvly small vectors, which creates a sparce matrixed with fast computetion of gradient. B·A approximates the update for original W, but expressed as a low-rank decomposition instead of a full update.
Use the training and validation set for trianing.
Loss function is set to Scale-Invariant Logarithmic Loss + gradient loss. 

### Loss charectaristics:
This function minimizes depth differences in log-space.
if the model predicts the correct geometric structure but scales the whole scene too large or too small, it isn't severely penalized due to the subtraction of a fraction of the squared mean log-error.
gradient loss shapends the predications.
