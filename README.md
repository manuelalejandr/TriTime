# TriTime

_TriTime: More Effective Triplets for Contrastive Learning in Semi-Supervised Time Series Classification_


## Overview
TriTime leverages unlabeled data by creating synthetic triplets consisting of an original reference time series and two transformed versions. Specifically, we use a conservative transformation to generate a positive sample and a destructive transformation to generate a negative sample. Our main contribution is a method that combines time series inversion and mixture to create hard negatives samples, that is, negative samples that are hard to discriminate from the positives. As a result, our approach enhances self-supervised learning, improving the model’s ability to learn time series representations from limited labeled data
## Runing Example

```
 python tritime_mixing.py
```
for Tritime Mixing 

```
 python tritime_mixup.py
```
for Tritime Mixup 

## Authors ✒️


* **Manuel Alejandro Goyo**
* **Ricardo Ñanculef**
* **Carlos Valle** 
