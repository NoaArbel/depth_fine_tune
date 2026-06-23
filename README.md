This repo is a small test project to fine-tune a small depth astimation model (depth anything).
The code uses lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320 data set to fine tune depth estimation for drone setup.

# Steps

## Evaluate pre-trained model
We run the model on 10% of the data (= test set).
Before computing metrics, predicated and GT depth are alighed to exclude scale and shift from the calculations and standardize everything to a unified measure. following https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide
The metric computed are:
* absRel - Absolute Relative Error - measuring how much the predicted distances differ from the true ones on average in percentage terms
* rmse - Root Mean Square Error
* deltas - Threashold accuracy. /delta1 measures the percentage of predicted pixels that differ from the true pixels by no more than 25%. 