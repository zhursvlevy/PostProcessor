# PostProcessor
A service for prediction rating and tags of posts on social networks. We trained multiple ML models such as Gradient boosting, LDA, BERT on pytorch frame work, we investigated target metrics (see docs/paper.pdf) on different text embedding  models and provide main [results](#main-results) on these tasks and source code for result reproducing. We got 0.2584 on $R^2$ at rating prediction task and 0.5211 on Recall@10 at tags prediction task. 

## Project structure
- data is dummy directories for research purpose (model weights saving, data storing)
- logs is directory for experiment results saving
- notebooks contains .ipynb files with explorational data analysis and some model pipelines. cleaning.ipynb is general data preprocessing pipeline. Rating_predict and topic_modelig contain experiments with regression and multilabel classification models and its configs.
- docs contains our comprehensional report about this project.
- scripts contains some helping scripts such embeddings extraction, general util functions for all project parts
- pytorch_rating_models are models for rich configuring bert training pipeline. The template is tooked from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
- pytorch_tagging_models are custom training pipeline for tags prediction task.


## Data
[Source](https://huggingface.co/datasets/IlyaGusev/pikabu). It Contains more than 279k rows of posts with its ratings (pluses and minuses), post tags and some additional information. To get source data run code below%
```console
wget -i scripts/datalinks.txt -P data/pikabu
```

## Download, process source data and lemmatize it (Now avalilable on Linux ubuntu platform only)
```console
make data_source && make lemmatize
```

## EDA, baselines some classical ML models
Are availabe at notebooks directory. 

## Train pytorch modesls
This project uses lightning for code structuring and hydra for flexible experiment configuring and tracking.
All models can be implemented in uniform structure. Models should be implemented in src/models/components. 
Training settings must be implemented in src/models. All data processing functional should be implemented in
src/data/components and lightning data modules must be implemented at src/data. All model configurations is set
in configs folder. You can configure model in .yaml files or set parameters in terminal:
```console
python src/train.py "hydra parameters overrides"
```
## Train pytorch models. Example
```console
python src/train.py \
    experiment=rate_prediction \
    model.net.hidden_dim=256 \
    model.optimizer.lr=1e-5
```
# Main results

## Rating prediction

We provide main results in table below. Best models is based on ruBERT embeddings. We provide links to source files to reproduce our results. More detailed information is available in docs/paper.pdf

|     Model     |       $R^2$   |  RMSE  |
| ------------- | ------------- | -------
|  [BTTFW](https://github.com/zhursvlevy/PostProcessor/tree/main/pytorch_rating_models/scripts/BTTFW.sh)        |     0.2584    | 0.2285  |
|  [CBTTW](https://github.com/zhursvlevy/PostProcessor/tree/main/notebooks/rating_predict/catboost_regressor.ipynb)         |     0.2531    | 0.2293  |
|  [BTFW](https://github.com/zhursvlevy/PostProcessor/tree/main/pytorch_rating_models/scripts/scripts/BTFW.sh)         |     0.2461    | 0.2304  |
|  [BW](https://github.com/zhursvlevy/PostProcessor/tree/main/pytorch_rating_models/scripts/BW.sh)           |     0.2411    | 0.2312  |
|  [BS](https://github.com/zhursvlevy/PostProcessor/tree/main/pytorch_rating_models/scripts/BR.sh)           | 0.0609 | 0.9972  |

1. BTTFW = finetuned BTTFW = **B**ERT with **T**itle prepending to the text markdown with **F**reezed weights after first two epochs with **W**ilson scrore targets.
2. CBTTW = CatBoost trained on finetuned bert embedding with **W**ilson scrore targets.
3. BTFW = finetuned **B**ERT with text markdown only and **F**reezed weights after first two epochs with **W**ilson scrore targets.
4. BW = end-to-end trained feed-forward regression network with freezed pretraied **B**ERT backbone with **W**ilson scrore targets.
5. BR = end-to-end trained feed-forward regression network with freezed pretraied **B**ERT backbone with Stantartized pluses and minuses.

## Tags prediction

We provide all results of tags prediciton task in tables down below. The tables indicate the values of the $recall@k$ metric.  
In first table you can find out results of implementing latent Dirichlet allocation for tags prediction.

| W-OT | WT | Manual |
| ------ |-----| ------|
| 0.0280 | 0.0268 | 0.1446 |

1. W-OT = LDA model without any information about basic distribution of tags over dataset.
2. WT = LDA model with information about distribution of tags as a-priori distribution.
3. Manual = WT model, but tags are manually mapped to topics, instead of simple instead of matching across distributions.

At second table we present results of solving tags prediction task as classical classification task. 

|     Model \ Embeddings    |       BoW   |  TF-IDF  | NGTF-IDF | RB | W2V |
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| LogReg  | 0.3041 | 0.2968 | 0.2909 | 0.2750 | - |
|    XGB | 0.3653 | 0.3649 |0.4063| 0.2561 | - |
|    SVM  | 0.3080 | 0.2990 | - | 0.1625 | - |
|    LSTM  | - | 0.4584 | - | - | 0.5211 |

More detailed information about results and training process you can find out at docs/paper.pdf.
