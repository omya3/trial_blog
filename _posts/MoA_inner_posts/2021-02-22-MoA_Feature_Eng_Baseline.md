---
layout: inner_post
title:  "Feature Engineering and Baseline"
date:   2021-02-22 17:15:08 +0530
category: ML_insider
---

## Feature Engineering Ideas

1) **Sequential relationship** : There might be sequential relationship between the gene expressions. To detect them we can train 1D-Cnn or LSTM based network on them.

2) **Row wise statistics** : We can compute row wise statistics like mean, variance, skewness etc.  of features.

3) **Clustering** : We can apply clustering over gene expressions, cell viability features and get cluster label to which each sample belongs.

4) **Spatial information** : There might be spatial relationship between the gene expressions. To detecct that we can form images of gene expressioons, cell viability features and train 2d Cnn on them.

All such feature engineering techniques will be performed parallely with modelling procedure, since we will be performing **MultilabelStratifiedKFold** cross validation strategy so each time our train and validation set will change.

## Base line model

A simple base line model could be one which predicts presence of each label with probaility equal to average number of times that label is activated in the Training samples. Code is as follows:


```python
# read in 
scored     = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
sample     = pd.read_csv('../input/lish-moa/sample_submission.csv')

# calculate
predictions = []
for target_name in list(scored)[1:]:
    rate = float(sum(scored[target_name])) / len(scored)
    predictions.append(rate)
predictions = np.array( [predictions] * len(sample) )

# write out
sample.iloc[:,1:] = predictions
sample.to_csv('submission.csv',index=False)
```

* The submission and score of base line model is as shown here <br>
[Click here to view the submission notebook](https://www.kaggle.com/sailoromkar/cs1-notebook-1-baseline-svm-lr-models?scriptVersionId=54195227)



<img src="https://i.ibb.co/C0BsnX7/Screenshot-2021-02-13-at-1-28-20-PM.png" width="1000px"/>

**Conclusion**: Henceforth, when we develop the models we would try to attain loss below the above mentioned scores.


|Blog part| 
|----------|
|1. [MoA problem definition link]({{ site.baseurl }}{% post_url /2021-02-22-MoA %})|
|----------|
|2. [EDA on LISH MoA dataset]({{ site.baseurl }}{% post_url /MoA_inner_posts/2021-02-22-MoA_EDA %})|
|-------------------|
|4. [ML techniques on MoA dataset]({{ site.baseurl }}{% post_url /MoA_inner_posts/2021-02-22-MoA_ML %})