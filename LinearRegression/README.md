# LINEAR REGRESSION
In this, the use of Linear Regression to predict a numerical value is demonstrated.
<br>
This is achieved by using the following language/tools:
1. The Python programming language
2. NumPy - a Python library used for scientific programming
3. Matplotlib - a Python library for creating visualizations
<br>

Let us consider a dummy dataset of 100 samples with one feature, X and a label y. The data may look like this:
| Label (y)      | Feature (x)    |
|----------------|----------------|
| 6.45015697     | 1.09762701     |
| 10.26106065    | 1.43037873     |
| 8.88269879     | 1.20552675     |
| 6.76342256     | 1.08976637     |
| 9.08712957     | 0.8473096      |

<img src=".\LR_initial-data-points.png" alt="Initial data points visualization" width="500">

Let us define our regression hypothesis as:
$$
\hat{y} = \theta_0 + \theta_1 \cdot x
$$
where,
- $\hat{y}$ = predicted label/output vector
- $x$ = input feature vector
- $\theta_1$ = weight/co-efficient for input feature $x$
- $\theta_0$ = bias term
<br>

Now, let us initialize the weight and bias with some random value, say $\theta_1$ = 0.40015721, $\theta_0$ = 1.76405235