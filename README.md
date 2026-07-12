This repo is a small test project to fine-tune a small depth estimation model (depth anything).
The code uses lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320 data set to fine tune depth estimation for drone setup.

# Steps

## divide the data
Split the data to train (70%) validation (15%) and test (15%)


## Evaluate pre-trained model
We run the model the test set.
Before computing metrics, predicted and GT depth are aligned to exclude scale and shift from the calculations and standardize everything to a unified measure. following https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide

### Metrics
* absRel - Absolute Relative Error - measuring how much the predicted distances differ from the true ones on average in percentage terms
* rmse - Root Mean Square Error
* deltas - Threashold accuracy. /delta1 measures the percentage of predicted pixels that differ from the true pixels by no more than 25%. 

## Training
LoRA adaption is used for this fine-tuning, to reduce the computation. Some of the layers will get an adaption. The adaption is a set of additional weights for the input. This additional matrix is computed based on 2 relatively small vectors, which creates a sparse matrix with fast computation of gradient. B·A approximates the update for original W, but expressed as a low-rank decomposition instead of a full update.
Use the training and validation set for training.
Loss function is set to Scale-Invariant Logarithmic Loss.

### Loss charectaristics:
This function minimizes depth differences in log-space.
if the model predicts the correct geometric structure but scales the whole scene too large or too small, it isn't severely penalized due to the subtraction of a fraction of the squared mean log-error.

## Experiment grid
In order to show the effects of learning parameters, experiment grid is created. Thus, each experiment (training) is done with different params.
Example:
```
EXPERIMENTS = [
         {
        "name": "lr1e-4_r16-small",
        "overrides": {"training": {"learning_rate": 1e-4, "epochs": 5, "batch_size": 4},
                      "lora":     {"r": 16, "lora_alpha": 32}},
    },
]
```

# Code structure
Most of the code is encapsulated in ./src folder in python scripts. The actual run is done via notebooks. This allows to use Kaggle or Colab with the same src.

train.py includes training related functions, and it's main is used to test the trianing on a sub set of images, to check the pipeline.

## notebooks
train_colab notebook is the main notebook. It contain loading data, spliting it, run original netwrok pre-training, than create experiments, run them and save all evaluations.

## tests
Current tests only cover loss functions

# Results

## Process
Starting with fine-tuning query and value parts of the attention module, learning rates 1e-4 and 1e-5, and 5 epochs, with variety of LoRA rank (8,16).
After establishing the rank 16 was better, added 10 epoch test to see how it will effect the training with LR of 1e-4.
The result was better, but still the improvement was not what we want. We want to see more % of pixels in delta1 (ie error TH below 25%).

Next, we increased LR a little to 3e-4 and 6e-4, still, 10 epochs training is better (unsurprisingly). 6e-4 LR is probably too high. Then we tested again with 3e-4 and (1) 10 epochs (2) rank of 32.

Then, consider to increase the rank of the LoRA (and thus increase the number of parameters to train) and also add other attention parts to the fine-tuning (keys, output).

### adding gradiernt loss


## Restuls Table

Name                            LR      LoRA r  Grad λ  Batch   abs_rel     rmse        delta1
-------------------------------------------------------------------------------------------------
lr1e-4_r16-10-epoch             0.0001  16      0.0     1       0.4971      20.0864     0.3116
lr1e-4_r16_e20                  0.0001  16      0.0     4       0.5111      20.9510     0.3107
lr1e-4_r16-10-epoch-q-v-k-grad  0.0001  16      0.1     1       0.5089      21.0529     0.3087
lr1e-4_r32_e10                  0.0001  32      0.0     4       0.5333      21.0680     0.2856
lr1e-4_r16-10-epoch-q-v-k       0.0001  16      0.0     1       0.5455      20.7905     0.2906
lr1e-4_r32_e10-grad             0.0001  32      0.1     4       0.5750      21.7667     0.2543
lr3e-4_r16-e5                   0.0003  16      0.0     1       0.5856      21.9730     0.2609
lr3e-4_r32-e5                   0.0003  32      0.0     1       0.7492      24.1234     0.2592
lr1e-4_r16-5-epoch              0.0001  16      0.0     1       0.5587      21.7914     0.2526
lr1e-4_r16_e20-grad             0.0001  16      0.2     8       0.5869      21.7335     0.2475
lr1e-4_r8-5-epoch               0.0001  8       0.0     1       0.5779      22.0196     0.2455
lr1e-5_r16-5-epoch              1e-05   16      0.0     1       0.6112      23.4979     0.2454
lr6e-4_r16-e5                   0.0006  16      0.0     1       0.9736      27.0095     0.2235
Baseline                        -       -       -       -       1.0883      28.1304     0.1298

## how do we know the training was good?
1. Metrics on test is better then before the trianing.
2. Check per epoch progress on training/val loss.

### training metrics vs baseline
*AbsRel*
The difference between true value and predicted value drop by half (on average).

*RMSE*
The sum of all the errors is lower, but no by much. 

*delta - 1.25*
Percentage of predicted pixels that differ from the true pixels by no more than 25% is two times higher -it mean that more pixels are now have lower error.

## visuals
* For lr3e-4_r32-e5_004668 the baseline depth is more "crisp" even if the max error is larger)
* For lr3e-4_r32-e5_003891 the baseline depth is better - more "crisp" and max error is smaller.

## Hybrid training
Since training with LoRA adaptor for the encoder of DepthAnything got us as far as delta1 = 0.31 and not so good visual depth map, the next step is to train the actual model. Trying to save on resources, start with partial fine tuning of parameters of the decoder.
Parameters break down:

Section                              Params       % of total
------------------------------------------------------------
backbone.embeddings                    752,640        3.0%
backbone.encoder (12 layers)        21,302,784       84.0%
backbone.layernorm                         768        0.0%
neck.reassemble_stage                1,678,560        6.6%
neck.convs                             414,720        1.6%
neck.fusion_stage (4 layers)         1,198,336        4.7%
head                                    27,745        0.1%
------------------------------------------------------------
TOTAL                               25,375,553      100.0%

neck + head combined:  3,319,361 params  (13.1%)
Attention query, key and values are a part of the layers in the backbone.encoder.

Each new experiment will define what part of the decoder will be unfrozen.
Example:
{
    "name": "lr1e-4_r16_hybrid_e10",
    "overrides": {
        "training": {"learning_rate": 1e-4, "epochs": 10, "batch_size": 4},
        "lora": {"r": 16, "lora_alpha": 32, "target_modules": ["query", "value"], "lambda_grad": 0.0},
        "decoder": {"unfreeze": ["head", "neck.convs"], "lr": 1e-5}
    },
}


