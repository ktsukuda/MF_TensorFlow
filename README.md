# MF_TensorFlow

Matrix Factorization with TensorFlow.

## Environment

- Python: 3.6
- TensorFlow: 2.2.0
- CUDA: 10.1
- Ubuntu: 18.04

## Dataset

[The Movielens 1M Dataset](http://grouplens.org/datasets/movielens/1m/) is used. The rating data is included in [data/ml-1m](https://github.com/ktsukuda/MF_TensorFlow/tree/master/data/ml-1m).

## Run the Codes

```bash
$ python MF_TensorFlow/main.py
```

## Details

For each user, the latest and the second latest rating are used as test and validation, respectively. The remaining ratings are used as training. The hyperparameters (batch_size, lr, latent_dim, l2_weight) are tuned by using the valudation data in terms of nDCG. See [config.ini](https://github.com/ktsukuda/MF_TensorFlow/blob/master/MF_TensorFlow/config.ini) about the range of each hyperparameter.

Although the original ratings range 1 to 5, all of them are converted to 1. That is, we use the binalized data where movies rated by users have score 1 while those not rated by users have score 0.

By running the code, hyperparameters are automatically tuned. After the training process, the best hyperparameters and HR/nDCG computed by using the test data are displayed.

Given a specific combination of hyperparameters, the corresponding training results are saved in `data/train_result/<hyperparameter combination>` (e.g., data/train_result/batch_size_512-lr_0.005-latent_dim_8-l2_reg_1e-07-epoch_3-n_negative_4-top_k_10). In the directory, model files and a json file (`epoch_data.json`) that describes information for each epoch are generated. The json file can be described as follows (epoch=3).

```json
[
    {
        "epoch": 0,
        "loss": 3645.9334130585194,
        "HR": 0.4816225165562914,
        "NDCG": 0.2670223724035898
    },
    {
        "epoch": 1,
        "loss": 3249.202230915427,
        "HR": 0.5198675496688742,
        "NDCG": 0.2887319064859957
    },
    {
        "epoch": 2,
        "loss": 3057.551134392619,
        "HR": 0.5708609271523178,
        "NDCG": 0.31983457600351684
    }
]
```
