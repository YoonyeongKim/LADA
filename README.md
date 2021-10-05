# LADA: Look-Ahead Data Acquisition via Augmentation for Deep Active Learning

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py --data Cifar10 --method LADA
```

## Evaluation

- Data will be downloaded to folder 'data'.
- Result will be recorded to folder 'Results'.

## Results

Our model achieves the following performance on active learning settings:

| Model name  | FashionMNIST  |      SVHN     |    CIFAR-10   |   CIFAR-100   |
| ----------- |-------------- | ------------- | ------------- | ------------- |
|    LADA     |     83.68%    |     75.72%    |     53.45%    |     46.92%    |
