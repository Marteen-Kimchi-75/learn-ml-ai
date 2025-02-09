# MULTINOMIAL LOGISTIC REGRESSION
This implementation demonstrates multiclass classification using the Iris dataset to predict flower species based on four features.
<br>
**Tools/Libraries Used**:
1. Python Programming Language
2. NumPy - Numerical computing
3. Pandas - Data manipulation
4. Matplotlib - Visualization

## DATASET PREPARATION
The Iris dataset contains 150 samples with 4 features and 3 target classes:
- Features: Sepal length, Sepal width, Petal length, Petal width
- Classes: setosa (0), versicolor (1), virginica (2)

**Sample Data**:
| sepal_length | sepal_width | petal_length | petal_width | species |
|--------------|-------------|--------------|-------------|---------|
| 5.1          | 3.5         | 1.4          | 0.2         | 0       |
| 7.0          | 3.2         | 4.7          | 1.4         | 1       |
| 6.3          | 3.3         | 6.0          | 2.5         | 2       |

**Preprocessing**:
- Mapped species names to numerical labels (0, 1, 2)
- One-hot encoded target labels
- Added bias term to features

## MODEL DEFINITION
The hypothesis function uses **softmax** for multiclass probability estimation:
$$
P(y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \quad \text{where} \quad z = X\theta
$$
where:
- $X$ = Feature matrix with bias term
- $\theta$ = Weight matrix (parameters to learn)
- $K$ = Number of classes (3 for Iris dataset)

$z = X\theta$ can also be written as: <br>

$\quad$ $z_k = \theta_{0k} + \theta_{1k} x_1 + \theta_{2k} x_2 + \theta_{3k} x_3 + \theta_{4k} x_4$

where, 
- $z_k$ = logit for class k
- $\theta_{0k}$ .. $\theta_{4k}$ = parameters for class k
- $x_0$ .. $x_4$ = input features ($x_0$ = 1)

## LOSS FUNCTION
Cross-Entropy Loss is used for optimization:
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(p_k^{(i)})
$$
where:
- $y_k^{(i)}$ = True label (one-hot encoded) of $i^{th}$ example
- $p_k^{(i)}$ = Predicted probability from softmax of $i^{th}$ example
- $m$ = Number of training examples

## TRAINING PROCESS
**Gradient Descent Update Rule**:
$$
\theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}
$$
where $\alpha$ = learning rate (0.1 in this implementation)

**Training Details**:
- Initial parameters: Zero-initialized weight matrix
- Iterations: 5000
- Cost reduction visualization:

<img src=".\LoR_Cost-over-iterations.png" alt="Cost Reduction Over Iterations" width="700">

## MODEL EVALUATION
**Final Results**:
- Final cost: 0.07234277606135048
- Learned parameters matrix shape: (5 features × 3 classes)
- Learned parameters matrix:
$$
\begin{bmatrix}
0.67828994 & 1.84547555 & -2.5237655 \\
1.46664342 & 0.9335388 & -2.40018222 \\
3.19615017 & 0.10940036 & -3.30555053 \\
-4.3558272 & -0.28680745 & 4.64263465 \\
-2.09922593 & -2.19738116 & 4.29660709
\end{bmatrix}
$$