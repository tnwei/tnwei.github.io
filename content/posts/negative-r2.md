---
title: "Explaining negative R-squared"
date: 2022-06-06T00:36:00+08:00
draft: false
summary: "Why and when does R-squared, the coefficient of determination, go below zero"
tags: ["statistics"]
math: true
---

When I first started out doing machine learning, I learnt that:

- $R^2$ is the coefficient of determination, a measure of how well is the data explained by the fitted model,
- $R^2$ is the square of the coefficient of correlation, $R$,
- $R$ is a quantity that ranges from 0 to 1

Therefore, $R^2$ should also range from 0 to 1.

Colour me surprised when the `r2_score` [implementation in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) returned negative scores. What gives?

{{< figure src="/images/r2-gru.jpeg" alt="Same info in the intro, but in Gru meme format" caption="If you glossed over the math by instinct, this meme is for you." align=center >}}

## The answer lies in the definition

$R^2$ is defined upon the basis that the total sum of squares of a fitted model is equal to the explained sum of squares plus the residual sum of squares, or: 

$$SS_{tot} = SS_{exp} + SS_{res} \\ ... \\ (1)$$

where:

- Total sum of squares ($SS_{tot}$) represent the total variation in data, measured by the sum of squares of the difference between expected and actual values,
- Explained sum of squares ($SS_{exp}$) represent the variation in data explained by the fitted model, and
- Residual sum of squares ($SS_{res}$) represent variation in data that is not explained by the fitted model.

$R^2$ itself is defined as follows:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \\ ... \\ (2)$$

Given these definitions, note that negative $R^2$ is only possible when the residual sum of squares ($SS_{res}$) exceeds the total sum of squares ($SS_{tot}$). As this is not mathematically possible, it can only mean that the explained sum of squares and residual sum of squares no longer add up to equal the total sum of squares. In other words, the equality $SS_{tot} = SS_{exp} + SS_{res} $ does not appear [^1] to be true.

How can this be?

## Because we evaluate models separately on train and test data

Following the above definitions, $SS_{tot}$ can be calculated using just the data itself, while $SS_{res}$ depends both on model predictions and the data. While we can use any arbitrary model to generate the predictions for scoring, we need to realize that the aforementioned equality is defined for _models trained on the same data_. Therefore, it doesn't necessarily hold true when we use test data to evaluate models built on train data!  There is no guarantee that the differences between a foreign model's predictions and the data is smaller than the variation within the data itself. 

We can demonstrate this empirically. The code below fits a couple of linear regression models on randomly generated data: 

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

for _ in range(20):
    data = np.random.normal(size=(200, 10))

    X_train = data[:160, :-1]
    X_test = data[160:, :-1]
    y_train = data[:160, -1]
    y_test = data[160:, -1]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    print(f"Train R2: {train_score:.3f}, Test R2: {test_score:.3f}")
```

Try as we might, the $R^2$ never drops below zero when the models are evaluated on train data. Here's what I got in STDOUT:

```
Train R2: 0.079, Test R2: -0.059
Train R2: 0.019, Test R2: -0.046
Train R2: 0.084, Test R2: -0.060
Train R2: 0.020, Test R2: -0.083
Train R2: 0.065, Test R2: -0.145
Train R2: 0.022, Test R2: 0.032
Train R2: 0.048, Test R2: 0.107
Train R2: 0.076, Test R2: -0.031
Train R2: 0.029, Test R2: 0.006
Train R2: 0.069, Test R2: -0.150
Train R2: 0.064, Test R2: -0.150
Train R2: 0.053, Test R2: 0.096
Train R2: 0.062, Test R2: 0.022
Train R2: 0.063, Test R2: 0.008
Train R2: 0.059, Test R2: -0.061
Train R2: 0.076, Test R2: -0.191
Train R2: 0.049, Test R2: 0.099
Train R2: 0.040, Test R2: -0.012
Train R2: 0.096, Test R2: -0.373
Train R2: 0.073, Test R2: 0.088
```

## So ... what about $R^2$ being the square of correlation?

It appears that $R^2 = R * R$ only under limited circumstances. Quoting the paragraph below from the [relevant Wikipedia page](https://en.wikipedia.org/wiki/Coefficient_of_determination):

> There are several definitions of $R^2$ that are only sometimes equivalent. One class of such cases includes that of simple linear regression where $r^2$ is used instead of $R^2$. When only an intercept is included, then $r^2$ is simply the square of the sample correlation coefficient (i.e., $r$) between the observed outcomes and the observed predictor values. If additional regressors are included, $R^2$ is the square of the coefficient of multiple correlation. In both such cases, the coefficient of determination normally ranges from 0 to 1. 

In short, $R^2$ is only the square of correlation if we happen to be (1) using linear regression models, and (2) are evaluating them on the same data they are fitted (as established previously).

## On the liberal use of $R^2$ outside the context of linear regression

The quoted Wikipedia paragraph lines up with my observation flipping through statistical texts:  $R^2$ is almost always introduced within the context of linear regression. That being said, the formulation of $R^2$ makes it universally defined for any arbitrary predictive model, regardless of statistical basis. It is used liberally by data scientists in regression tasks, and is even the default metric for regression models in sklearn. Is it right for us to use $R^2$ so freely outside its original context?

Honestly, I don't know. On one hand, it clearly has a lot of utility as a metric, which led to its widespread adoption by data scientists in the first place. On the other hand, you can find [discussions](https://stats.stackexchange.com/questions/547863/heres-why-you-can-hopefully-use-r2-for-non-linear-models-why-not) [like](https://blog.minitab.com/en/adventures-in-statistics-2/why-is-there-no-r-squared-for-nonlinear-regression) [these](https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/) online that caution against using $R^2$ for non-linear regression. It does seem to me that from a statistics perspective, it is important for $R^2$ to be calculated under the right conditions such that its properties can be utilized for further analysis. I take my observed relative lack of discourse about $R^2$ within data science circles to mean that from a data science perspective, $R^2$ doesn't mean more than being a performance metric like MSE or MAE.

Personally, I think we are good with using $R^2$, as long as we understand it enough to know what _not_ to do with it.

## Wrapping up

To summarize, we should expect $R^2$ to be bounded between zero and one only if a linear regression model is fit, and it is evaluated on the same data it is fitted on. Else, the definition of $R^2$ being $1 - \frac{SS_{res}}{SS_{tot}}$ can lead to negative values.



[^1]: Being specific with my choice of words here.
