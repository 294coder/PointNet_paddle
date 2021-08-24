# PointNet_paddle

implementation of PointNet using Paddle(classification only).

## Paper

![image-20210825055752643](294coder/PointNet_paddle/pic/image-20210825055752643.png)

[here](https://arxiv.org/abs/1612.00593)

## Environment

sklearn>=0.24.0

paddle>=0.2.0

tqdm

## Performance

use AdamW as optimizer with learn rate 0.0001

|              Model              | Accuracy |
| :-----------------------------: | :------: |
|        PointNet(Offical)        |   89.2   |
|        PointNet(PyTorch)        |   90.6   |
| PointNet(Paddle without normal) |   91.2   |
|  PointNet(Paddle with normal)   |          |

## Dataset

### ModelNet40

Download datasets [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)

 train : 9843
 test : 2468

## Quickstart
