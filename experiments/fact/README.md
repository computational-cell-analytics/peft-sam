# FacT
This folder contains experiments that aim to find optimal hyperparameters for the FacT method.
## Dropout Experiments
It is tested whether or not the FacT method profits from dropout layers. 

### Inference Results

|    |dropout factor|      ais |      amg |       ip |       ib |    point |      box |
|---:|:------------|---------:|---------:|---------:|---------:|---------:|---------:|
|  0 |        0.25 | 0.386388 | 0.26497  | 0.783651 | 0.82816  | 0.40846  | 0.64648  |
|  1 |        None | 0.383379 | 0.256524 | 0.779622 | 0.822782 | 0.400679 | 0.64233  |
|  2 |        0.1  | 0.38944  | 0.265567 | 0.787556 | 0.830561 | 0.413252 | 0.649592 |
|  3 |        0.5  | 0.385142 | 0.2589   | 0.77919  | 0.821817 | 0.397363 | 0.640852 |
### Training Time

|    | dropout factor    |   Best Training Time |   Latest Training Time |   Time per Iteration |
|---:|:------------------|---------------------:|-----------------------:|---------------------:|
|  0 |        0.25       |              85696.6 |               105393   |              2.10787 |
|  1 |        None       |              61608.4 |                89815.9 |              1.79632 |
|  2 |        0.1        |             104977   |               104977   |              2.09954 |
|  3 |        0.5        |              78919.3 |               105452   |              2.10905 |

## Rank Study