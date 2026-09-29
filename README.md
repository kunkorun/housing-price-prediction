# 🏠 House Prices — Advanced Regression

This was my second Kaggle Machine Learning project.

The goal was to predict house prices based on different characteristics of residential properties.

**Kaggle RMSLE: 0.12444**

\---

## 🎯 Goal

In this project I wanted to move beyond my first classification project and work with a regression problem.

I focused on:

* understanding a larger and more complex dataset;
* performing EDA;
* handling missing values;
* creating new features;
* transforming the target;
* comparing different models;
* using cross-validation;
* analysing which changes actually improved the result.

\---

## 🔍 Exploratory Data Analysis

I started by analysing the structure of the dataset, missing values, distributions and relationships between the features and the target.

Some of the features required more attention because they contained many missing values or had strongly skewed distributions.

I used visualizations during the analysis to better understand these patterns before deciding how to preprocess the data.

\---

## 🧹 Preprocessing

I experimented with different ways of handling:

* missing values;
* categorical features;
* numerical features;
* skewed distributions;
* potentially unhelpful columns.

One important part of the project was trying to make preprocessing decisions based on the meaning of the features rather than applying the same approach to every column.

\---

## 🛠️ Feature Engineering

I created and tested additional features based on the information already available in the dataset.

The goal was to give the models features that could represent useful relationships in the housing data more clearly.

I also paid attention to whether a transformation could introduce data leakage or behave differently between training and validation data.

\---

## 📈 Target Transformation

House prices are strongly skewed, so I experimented with transforming the target variable.

This helped make the target distribution easier for the models to work with and was especially relevant because the competition evaluates predictions using RMSLE.

\---

## 🤖 Models \& Validation

I compared several regression models and used cross-validation to evaluate them.

I tried to use validation not just to find the highest score, but to understand whether an improvement was actually consistent.

The general workflow was:

```text
Baseline
   ↓
Experiment
   ↓
Cross-validation
   ↓
Compare with baseline
   ↓
Keep or reject the change
```

\---

## 📊 Evaluation

The competition uses **Root Mean Squared Logarithmic Error (RMSLE)**.

My final Kaggle score:

**0.12444**

For this metric, a lower score is better.

\---

## 🔎 Experiments

Throughout the project I tested different ideas around:

* preprocessing;
* feature engineering;
* target transformation;
* model selection;
* validation.

I tried to keep track of the changes and compare them against a baseline instead of changing many things at once.

One important lesson from this project was that a change that improves the local validation score is not automatically a good improvement unless the validation setup itself is reliable.

\---

## ⚠️ Validation \& Leakage

While reviewing the project afterwards, I found an important point that I would improve in the future.

Some preprocessing was performed before cross-validation, which means that the validation process was not completely isolated from preprocessing.

This was a useful lesson for me:

> preprocessing should be fitted only on the training part of each fold whenever possible.

I would move this logic into the validation pipeline in a future version of the project.

\---

## 💡 What I Learned

This project taught me more about:

* regression problems;
* cross-validation;
* target transformations;
* feature engineering;
* skewed distributions;
* the importance of a reliable validation strategy;
* the difference between improving a score and improving a real experiment.

Compared with my first project, I started paying more attention to **why** a change improved or worsened the model instead of only looking at the final score.

\---

## 🚧 What I Would Improve

If I returned to this project now, I would:

* make the preprocessing fully leakage-free inside the CV pipeline;
* organise experiments more systematically;
* perform more detailed error analysis;
* compare the stability of different models across folds.

\---

## ▶️ How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook and run the analysis from the beginning.

\---

## 🔗 Links

* **Kaggle:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* **My Kaggle Profile:** [kunkorun](https://www.kaggle.com/kunkorun)

