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

<img src=".\LR_initial-data-points.png" alt="Initial data points visualization" width="700">

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

<br>
Predicting with the current model gives us the following regression line:
<img src=".\LR_Initial-Regression-Line.png" alt="Initial regression line" width="700">

## Training the model
To get reasonable predictions, the model is trained using an algorithm known as **Gradient Descent**.
<br>

But first, a loss function needs to be selected. In this case, the loss function selected is **Mean Squared Error (MSE)**, which is defined as:
$$
MSE = \frac{1}{2m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
$$
where, 
- $\hat{y}_i$ = predicted output for $i^{th}$ example
- $y_i$ = true label for $i^{th}$ example
- $m$ = number of training examples

The curve of this loss function as of $\theta_0$ and $\theta_1$ might look like:
<img src=".\LR_MSE_Parabolic_Curve.png" alt="MSE Curve" width="600">

Now, we implement gradient descent to update the weight and bias in the following manner:
$$
\theta_0 := \theta_0 - \alpha \frac{\partial J(\theta)}{\partial \theta_0}
$$

$$
\theta_1 := \theta_1 - \alpha \frac{\partial J(\theta)}{\partial \theta_1}
$$
where,
- $\frac{\partial J(\theta)}{\partial \theta_1}$ = Partial derivative of the loss function (MSE) w.r.t. $\theta_1$
- $\frac{\partial J(\theta)}{\partial \theta_0}$ = Partial derivative of the loss function (MSE) w.r.t. $\theta_0$
- $\alpha$ = learning rate

Gradient descent is applied for 1000 iterations and the change in loss over each iteration looks like this:
<img src=".\LR_Cost-Reduction.png" alt="Cost reduction over iterations" width="700">

The final regression line after training the model looks like this:
<img src=".\LR_Final-Regression-Line.png" alt="Final Regression Line" width="700">

And the final parameters are $\theta_1$ = 2.9685 and $\theta_0$ = 4.222 with a loss of 0.4962