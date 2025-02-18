# TriTime

_TriTime: Cost-effective Triplets for Self-supervised Time Series Classification_


## Overview
Time series classification is critical in various real-world applications such as gesture recognition, ECG classification, and fault detection. Recently, deep learning has achieved significant advancements in the task, yet these models often require extensive labeled datasets, which are costly and time-consuming to collect. To address this challenge, we propose TriTime, a novel self-supervised framework based on triplets for semi-supervised time series classification. 

TriTime leverages unlabeled data by creating synthetic triplets consisting of an original reference time series and two transformed versions. Specifically, we use a conservative transformation to generate a positive sample and a destructive transformation to generate a negative sample. We evaluate three different methods to generate the negative sample in the triplet. Our first method relies on time series inversion, an operation introduced as a practical pretext task in previous self-supervised methods. Our second method combines time series inversion with a mixing operation in the time series space to create more effective negatives, that is, negative samples that are harder to discriminate from the positives. Our third and last method implements the latter idea in the latent space learned by the neural net.  

Experiments on multiple time series datasets demonstrate that time series inversion alone is a competitive method compared to the current state of the art. However, the negatives generated by combining inversion and mixing significantly outperform existing methods in most cases, even in scenarios with limited labeled data.
## Runing Example

```
 python tritime.py
```
for Tritime 

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
