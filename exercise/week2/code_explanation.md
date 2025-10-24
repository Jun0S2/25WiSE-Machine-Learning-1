# Maximum Likelihood Parameter Estimation

# Probability Distribution

A model that mathematically describes how data is generated:

- Tossing a coin: Bernoulli
- Height, weight: Gaussian
- When there are many outliers: Cauchy

---

# Cauchy Probability

$$
p(x|\theta) = \frac{1}{\pi(1+(x-\theta)^2)}
$$

- x: observed data
- θ : hypothesis representing the true state of the world (like a condition)
- hypothesis representing the true state of the world (like a condition)
- Given parameter θ the probability of observing data x

### Gaussian vs Cauchy

```python
   height
    |
 1.0|            Gaussian (tame)
    |          ___
0.5|         /    \
    |       /      \
0.2|-------/--------\---------  Cauchy (heavy tails)
    |_____/          \______
       -3  -2  -1   0   1   2   3  --> x

```

## Likelyhood

$$
L(\theta) = p(D|\theta) = \prod_ip(x_i|\theta)
$$

Multiply the probabilities of all data points, then find the θ that maximizes it — this θ is the one that best explains the data.

# MLE

### Step 1 ) Apply log

$$
\log L(\theta) = \sum_i \log p(x_i|\theta)
$$

### Step 2) MLE

$$
\hat\theta _{MLE} = \arg \max_{\theta}\log L (\theta)
$$

- argument of the maximum : 가장 큰 값을 만드는 theta를 찾아라
- max theta : theta를 바꿔가며 가장 큰 값을 찾아라
- log L(theta) : log를 취한 likelyhood
- L(theta) : 데이터가 theta 일때 나올 확률

### Example )

$$
f(\theta) = -(\theta - 3)^2 +5
$$

- max f(theta) = 5 ← 최대 높이
- arg max f(theta) = 3 ← 그 높이를 만드는 theta 값

---

## Code

- x : data set `numpy.array([1.2,0.7,…])`
- Theta_list : `numpy.array([-2,-1,0])`

### Flow

1. Cacluate  p(xi∣θ)p(xi∣θ) for each θ
2. ∑ilog⁡p(xi∣θ)∑ilogp(xi∣θ)
3. save in array for all log-likelihood

---

Implement a function that takes a dataset `D` as input (given as one-dimensional array of numbers) and a list of candidate parameters `θ` (also given as a one-dimensional array), and returns a one-dimensional array containing the log-likelihood w.r.t. the dataset `D` for each parameter `θ`

```python
def ll(D,THETA):

    # --------------------------------------
    # returns a one-dimensional array containing the
    # log-likelihood with respect to the dataset D for each parameter
    # log-likelihood : log L(theta) = Sigma log p (xi | theta)
    # so output should be [log L (theta1), Log L (theta 2)]....
    # --------------------------------------
    # import solutions; return solutions.ll(D,THETA)
    # --------------------------------------
    log_likelihoods = []  # 결과 저장용 리스트

    for t in THETA:
        p = pdf(D, t)               # 각 데이터 x_i 에 대한 확률 p(x_i | theta)
        logL = numpy.sum(numpy.log(p))    # log likelihood = sum(log(p))
        log_likelihoods.append(logL)

    return numpy.array(log_likelihoods)
```

# Building a Classifier

### Classifiers?

literally decide - `new data` belongs to whih `class`

### Maximum Likelihood based Classifier

Data per Class

- D1, D2

Possibility Model (Cauchy in this exercise)

- $P(x|\theta) = \frac{1}{\pi}\frac{1}{1+(x-\theta)^2}$

Prior

- $P(w_1), P(w_2)$
- use if you already know the possibility

## Flows

```python
D1 -> fit -> θ1_hat  --->\
                            \
                             --> g(x) -> decide class
D2 -> fit -> θ2_hat  --->/
```

### 1. ML 추정 (fit)

### Class 1 Data

$$
\hat\theta _{1} = \arg \max_{\theta}\sum_{x\in D_1} \log P (x|\theta)
$$

### Class 2 Data

$$
\hat\theta _{2} = \arg \max_{\theta}\sum_{x\in D_2} \log P (x|\theta)
$$

## 2. Discriminant Function g(x)

$$
g(x)=\log P(x∣\hatθ_1)−\log P(x∣\hatθ_2)+\log P(ω_1)−\log P(ω_2)
$$

- $\log P(x| \hat \theta_1 )$
  - probability that x will be observed in class 1 (theta 1)
  - bigger the value, the possibility inc.
- $-\log P(x| \hat \theta_2)$
  - probability that x will be observed in class 2 (theta 2)
  - smaller the value, the possibility inc.
- $\log P (w_1) - \log P (w_2)$
  - prior possibility applied
  - ex : if theres more class 1, then it will be class 1

rules :

- g(x) > 0 : class 1
- g(x) < 0 : class 2

So, it is a function that decide where `x` belongs

```python
class MLClassifier:
    # Compute (D1, D2) using ML
    def fit(self,THETA,D1,D2):
        # --------------------------------------
        # D1 = numpy.array([ 2.803, -1.563, -0.853,  2.212, -0.334,  2.503])
        # D2 = numpy.array([-4.510, -3.316, -3.050, -3.108, -2.315])
        # class Data :
        # theta = argmax_{theta} SUM (likelihood)
        # ll(D,THETA) returns numpy.array(log_likelihoods) so we can use this
        # --------------------------------------

        self.theta1 = THETA[numpy.argmax(ll(D1,THETA))]
        self.theta2 = THETA[numpy.argmax(ll(D2,THETA))]

        return self.theta1,self.theta2


	def predict(self, X, p1, p2):
	    # --------------------------------------
	    # Compute discriminant function g(x) over the new Data X
	    # p1, p2 : prior probability of each class 1,2
	    # return: g(x) vector
	    # g(x) > 0 -> class 1, g(x) < 0 -> class 2
	    # --------------------------------------
	    p_x_t1 = pdf(X, self.theta1)
	    p_x_t2 = pdf(X, self.theta2)
	    prior_diff = numpy.log(p1) - numpy.log(p2)

	    g_x = numpy.log(p_x_t1) - numpy.log(p_x_t2) + prior_diff
	    return g_x

```

# Bayes Parameter Estimation

## MLE v.s. Bayesian Estimation

### MLE (Maximum Likelihood Estimation)

$$
\hat{\theta}_{ML} = \arg\max\theta p(D|\theta)
$$

---

### Bayesian Parameter Estimation

$$
\hat{\theta}_{MAP} = \arg\max\theta p(\theta|D)
$$

---

## Prior

Data model is same as MLE →p(x|θ) - pdf (X,θ)

But now, we multiply `prior` to calculate `posterior`

$$
p(\theta) = \frac{1}{10\pi} \cdot \frac{1}{1+(\theta/10)^2}
$$

- Cauchy (p=0, scale =10)

## Posterior Formula

$D = \set {x_1,...,x_N}$

$$
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{\int p(D|\theta)p(\theta)d\theta}
$$

- lieklihood : $p(\theta|D) = \prod_ip(x_i|\theta)$
- prior : $p(\theta|D$) (cauchy)
- evidence (norm constant) : ${\int p(D|\theta)p(\theta)d\theta}$

```python
def prior(THETA):

    # --------------------------------------
    # we return p(\theta) = \frac{1}{10\pi} \cdot \frac{1}{1+(\theta/10)^2}
    # --------------------------------------
    return (1.0 / (10.0 * numpy.pi)) * (1.0 / (1 + (THETA / 10.0)**2))

def posterior(D,THETA):

    # --------------------------------------
    # we return p(\theta|D) = \frac{p(D|\theta)p(\theta)}{\int p(D|\theta)p(\theta)d\theta}
    # p(\theta|D) : likelihood
    # p(theta) : prior
    # denominator : normalization constant
    # https://numpy.org/doc/1.25/reference/generated/numpy.trapz.html
    # --------------------------------------
    # Note, that ll returns LOG likelihoods, so you need to exponentiate them first
    likelihood = numpy.exp(ll(D,THETA))  # p(D | theta)
    prior_probs = prior(THETA)  # p(theta)
    norm_constant = numpy.trapz(likelihood* prior_probs, THETA)  # normalization constant
    return likelihood*prior_probs / norm_constant
```

# Building a classifier

```python
        θ ~ p(θ|D1)
          |
          v
   integrate p(x|θ)p(θ|D1)
          |
        p(x|D1)   --- Bayesian marginalization
```

In MLE :

$$
g(x)=\log P(x∣\hatθ_1)−\log P(x∣\hatθ_2)+\log P(ω_1)−\log P(ω_2)
$$

But in Bayesian, $\theta$ is uncertain

## Bayesian Classifier

### Dataset-conditioned data density

Posterior from data D from class j

$$
p(\theta|D_j)
$$

And the new margin possibility for the data x:

$$
p(x|D_j) = \int p(x|\theta)p(\theta|D_j)d\theta
$$

- because $\theta$ is uncertain, we need weighted average
- $\theta$ is not constant so we need to integral posterior to get likelihood

### New Discriminant Function

$$
h(x) = \log P(x|D_1) - \log P(x|D_2) + \log P(w_1) - \log P (w_2)
$$

- Instead of $P(x|\hat \theta_j)$ , we use $p(x|D_j)$ which is dataset-conditioned density

and the same rules apply :

- g(x) > 0 : class 1
- g(x) < 0 : class 2

```python
px_D1 = []
for x in X:
    px = np.trapz(pdf(x, THETA) * self.post1, THETA)
    px_D1.append(px)
px_D1 = np.array(px_D1)

is same as
pdf(X[:, None], THETA[None, :])

```
