---
marp: true
paginate: true
theme: gaia
class: invert
size: 4:3
math: mathjx
---
# Classification Problems

---
What do we mean by quantitative response  variables  and qualitative response variables. Make a distinction between them ?

What is classification and how do we classify the observations to different categories or classes ?

---
List out the most widely used classifiers ?
1. logistic regression
2. linear discriminant analysis
3. quadratic discriminant analysis
4. naive bayes
5. K-nearest neighbors
6. generalized linear model : poission regression
7. generalized additive model : trees, random forests and boosting
8. Kernel based models : support vector machines

---
Identify the problems and it's responses
1. Data : $D = \{(x_i, y_i) : 1 \le i \le n\}$ we divide the our data into training, testing and validation sets with a ratio of 8: 1.5 : 0.5.
2. Aim is to build the classification model that can be trained on the dataset $D_{training}$ and being able to well generalized enough to make prediction on the useen data i.e $Y = f(X_{train})$

Note : It's better to estimate the probability of claim that feature belongs to some category rather than 

---
Can we use Linear Regression Models for Classification Tasks ?
> yes, but it's not better to use it for classification tasks because sometimes it make produce the outcome out of box. eg : In binary classification it may produce sometime less than zero or greater than zero. Hence we should't try out the linear regression model

Note : Even different encoding to the categorical variable can change the model prediction if we apply linear regression. Linear regression is equivalent to the linear disciminant analysis.

> Can we use the Logistic Regression for Classification tasks ?
> Yes, because it models the probability that category $y \in Y$. eg : Default loan prediction dataset, logistic regression can predict the probability of being default.