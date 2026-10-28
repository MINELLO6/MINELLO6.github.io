---
layout: post
title: "M04 - Gaussian Identities and Matrix Calculus"
date: 2026-10-27
mathjax: true
comments: true
---

# Gaussian Identities and Matrix Calculus

**Prerequisites:** M01 — Probability & Information Theory; M02 — Statistical Inference; M03 — Variational Inference. Basic linear algebra and multivariable calculus.

**Aim:** Develop the Gaussian identities and matrix calculus needed to understand probabilistic inference, VAEs, and diffusion models. In particular, derive conditional Gaussian distributions, products of Gaussian densities, Gaussian KL divergence, and the mathematical tools underlying DDPM.

---

## 1. Why Gaussian Distributions?

Gaussian distributions appear throughout probabilistic machine learning because they are mathematically tractable.

Three properties are particularly important:

1. Affine transformations of Gaussian random variables remain Gaussian.
2. Marginal and conditional distributions of jointly Gaussian variables are Gaussian.
3. Products of Gaussian densities, when viewed as functions of the same variable, are proportional to Gaussian densities.

These properties make it possible to derive many probabilistic inference equations analytically.

For example, the forward process of DDPM is defined by

$$
q(x_t\mid x_{t-1})
=
\mathcal N(
x_t;\sqrt{\alpha_t}x_{t-1},
\beta_t I
),
$$

where

$$
\alpha_t=1-\beta_t.
$$

Because the transition is Gaussian, we can derive

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
)
}
$$

without explicitly simulating all intermediate steps.

Furthermore, Gaussian identities allow us to calculate the conditional distribution

$$
q(x_{t-1}\mid x_t,x_0).
$$

This conditional distribution will play an important role in the DDPM variational objective.

Before studying diffusion models, we develop the mathematical tools needed to derive these results.

---

## 2. Multivariate Gaussian Distributions

### 2.1 Definition

A $d$-dimensional Gaussian random vector has distribution

$$
X\sim\mathcal N(\mu,\Sigma),
$$

where

$$
\mu\in\mathbb R^d
$$

is the mean vector, and

$$
\Sigma\in\mathbb R^{d\times d}
$$

is the covariance matrix.

For a non-degenerate Gaussian, $\Sigma$ must be symmetric and positive definite.

Its probability density is

$$
\boxed{
p(x)
=
\frac{1}{
(2\pi)^{d/2}|\Sigma|^{1/2}
}
\exp\left[
-\frac12
(x-\mu)^\top
\Sigma^{-1}
(x-\mu)
\right].
}
$$

The log-density is

$$
\boxed{
\begin{aligned}
\log p(x)
={}&-\frac d2\log(2\pi)\\
&-\frac12\log|\Sigma|\\
&-\frac12(x-\mu)^\top
\Sigma^{-1}(x-\mu).
\end{aligned}
}
$$

The three terms have distinct roles:

- $-\frac d2\log(2\pi)$ is a normalization constant.
- $-\frac12\log|\Sigma|$ accounts for the volume scale of the distribution.
- The quadratic term measures distance from the mean relative to the covariance structure.

### 2.2 Mahalanobis Distance

Define the squared Mahalanobis distance:

$$
d_M^2(x,\mu)
=
(x-\mu)^\top
\Sigma^{-1}(x-\mu).
$$

Unlike Euclidean distance, Mahalanobis distance accounts for correlations and different variances across dimensions.

When

$$
\Sigma=I,
$$

we recover

$$
d_M^2(x,\mu)=\|x-\mu\|_2^2.
$$

When

$$
\Sigma=\sigma^2I,
$$

we obtain

$$
d_M^2(x,\mu)
=
\frac{\|x-\mu\|_2^2}{\sigma^2}.
$$

Thus, Gaussian log-likelihood is closely related to squared-error objectives.

### 2.3 The Covariance Matrix

The covariance matrix is defined as

$$
\Sigma
=
\mathbb E[
(X-\mu)(X-\mu)^\top
].
$$

Its diagonal entries represent variances, while off-diagonal entries represent covariances.

For example,

$$
\Sigma=
\begin{pmatrix}
1 & 0.8\\
0.8 & 1
\end{pmatrix}
$$

describes two positively correlated random variables.

A diagonal covariance matrix assumes zero pairwise correlations.

For a jointly Gaussian vector, zero covariance also implies independence.

However, this implication does not hold for arbitrary distributions.

---

## 3. Affine Transformations and Gaussian Addition

### 3.1 Affine Transformations

Suppose

$$
X\sim\mathcal N(\mu,\Sigma)
$$

and define

$$
Y=AX+b.
$$

Then

$$
\boxed{
Y\sim
\mathcal N(
A\mu+b,
A\Sigma A^\top
).
}
$$

To derive this, first calculate the expectation:

$$
\begin{aligned}
\mathbb E[Y]
&=\mathbb E[AX+b]\\
&=A\mathbb E[X]+b\\
&=A\mu+b.
\end{aligned}
$$

The covariance is

$$
\begin{aligned}
\operatorname{Cov}(Y)
&=\mathbb E[
(Y-\mathbb E[Y])
(Y-\mathbb E[Y])^\top
]\\
&=\mathbb E[
A(X-\mu)(X-\mu)^\top A^\top
]\\
&=A\Sigma A^\top.
\end{aligned}
$$

Because an affine transformation preserves Gaussianity, these two quantities completely characterize the transformed distribution.

The resulting Gaussian may be degenerate if $A\Sigma A^\top$ is singular.

### 3.2 Adding Independent Gaussian Variables

Suppose

$$
X\sim\mathcal N(\mu_X,\Sigma_X)
$$

and

$$
Y\sim\mathcal N(\mu_Y,\Sigma_Y),
$$

with $X$ and $Y$ independent.

Then

$$
\boxed{
X+Y
\sim
\mathcal N(
\mu_X+\mu_Y,
\Sigma_X+\Sigma_Y
).
}
$$

More generally,

$$
\boxed{
aX+bY
\sim
\mathcal N(
a\mu_X+b\mu_Y,
a^2\Sigma_X+b^2\Sigma_Y
)
}
$$

for scalar coefficients $a,b$.

Independence is important: otherwise, cross-covariance terms must be included.

### 3.3 Application: Adding Gaussian Noise

Let

$$
X\sim\mathcal N(\mu,\Sigma)
$$

and

$$
\epsilon\sim\mathcal N(0,I),
$$

independently.

Consider

$$
Y=\sqrt\alpha X+\sqrt\beta\epsilon.
$$

Using the previous identities,

$$
\boxed{
Y\sim
\mathcal N(
\sqrt\alpha\mu,
\alpha\Sigma+\beta I
).
}
$$

If $X=x$ is treated as fixed, then

$$
\boxed{
Y\mid X=x
\sim
\mathcal N(
\sqrt\alpha x,
\beta I
).
}
$$

This distinction matters:

- $p(y\mid x)$ describes the distribution conditional on a fixed input.
- $p(y)$ averages over the randomness of $X$.

The DDPM forward process repeatedly applies the conditional Gaussian transformation.

---

## 4. Conditional Gaussian Distributions

This is one of the most important mathematical results for diffusion models.

### 4.1 Partitioning a Joint Gaussian

Suppose

$$
\begin{pmatrix}
X\\Y
\end{pmatrix}
\sim
\mathcal N
\left(
\begin{pmatrix}
\mu_X\\\mu_Y
\end{pmatrix},
\begin{pmatrix}
\Sigma_{XX}&\Sigma_{XY}\\
\Sigma_{YX}&\Sigma_{YY}
\end{pmatrix}
\right).
$$

Assume $\Sigma_{YY}$ is invertible.

The marginal distributions are

$$
X\sim\mathcal N(\mu_X,\Sigma_{XX})
$$

and

$$
Y\sim\mathcal N(\mu_Y,\Sigma_{YY}).
$$

Our goal is to find the conditional distribution

$$
p(X\mid Y=y).
$$

The result is

$$
\boxed{
X\mid Y=y
\sim
\mathcal N(
\mu_{X\mid y},
\Sigma_{X\mid Y}
)
}
$$

where

$$
\boxed{
\mu_{X\mid y}
=
\mu_X+
\Sigma_{XY}\Sigma_{YY}^{-1}
(y-\mu_Y)
}
$$

and

$$
\boxed{
\Sigma_{X\mid Y}
=
\Sigma_{XX}
-
\Sigma_{XY}
\Sigma_{YY}^{-1}
\Sigma_{YX}.
}
$$

The covariance expression is also called a Schur complement.

### 4.2 Derivation Using Gaussian Residuals

Define

$$
B=\Sigma_{XY}\Sigma_{YY}^{-1}
$$

and construct the residual

$$
R=X-\mu_X-B(Y-\mu_Y).
$$

Its expectation is

$$
\mathbb E[R]=0.
$$

Now calculate its covariance with $Y$:

$$
\begin{aligned}
\operatorname{Cov}(R,Y)
&=\operatorname{Cov}(X,Y)
-B\operatorname{Cov}(Y,Y)\\
&=\Sigma_{XY}-B\Sigma_{YY}\\
&=\Sigma_{XY}
-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YY}\\
&=0.
\end{aligned}
$$

Because $R$ and $Y$ are jointly Gaussian, zero covariance implies independence.

Rearranging the definition of $R$,

$$
X=\mu_X+B(Y-\mu_Y)+R.
$$

Conditioning on $Y=y$ gives

$$
X\mid Y=y
=
\mu_X+B(y-\mu_Y)+R,
$$

where the distribution of $R$ is unchanged because $R$ is independent of $Y$.

Therefore,

$$
\mathbb E[X\mid Y=y]
=
\mu_X+B(y-\mu_Y).
$$

Substituting the definition of $B$,

$$
\boxed{
\mathbb E[X\mid Y=y]
=
\mu_X+
\Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y).
}
$$

The covariance of the residual is

$$
\begin{aligned}
\operatorname{Cov}(R)
={}&\Sigma_{XX}
-B\Sigma_{YX}\\
&-\Sigma_{XY}B^\top
+B\Sigma_{YY}B^\top\\
={}&\Sigma_{XX}
-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.
\end{aligned}
$$

Hence,

$$
\boxed{
\operatorname{Cov}(X\mid Y)
=
\Sigma_{XX}
-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.
}
$$

### 4.3 Intuition

The conditional mean has the form

$$
\text{Prior mean}
+
\text{Correction}.
$$

The correction is

$$
\Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y).
$$

It uses the difference between the observed value $y$ and its expected value to update our estimate of $X$.

The conditional covariance is

$$
\text{Prior covariance}
-
\text{Uncertainty reduction}.
$$

Observing a correlated variable generally reduces uncertainty.

### 4.4 Example: Bivariate Gaussian

Suppose

$$
\begin{pmatrix}X\\Y\end{pmatrix}
\sim
\mathcal N
\left(
\begin{pmatrix}0\\0\end{pmatrix},
\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}
\right),
$$

where $|\rho|<1$.

Then

$$
\boxed{
X\mid Y=y
\sim
\mathcal N(
\rho y,
1-\rho^2
).
}
$$

When $\rho=0$, observing $Y$ provides no information about $X$.

When $|\rho|$ approaches $1$, the conditional variance approaches zero.

This illustrates how correlation affects posterior uncertainty.

---

## 5. Products of Gaussian Densities

Conditional Gaussian formulas are useful when a joint Gaussian distribution is available.

Another useful technique is to multiply Gaussian densities directly.

### 5.1 The Precision Parameterization

Consider

$$
p_1(x)=\mathcal N(x;\mu_1,\Sigma_1)
$$

and

$$
p_2(x)=\mathcal N(x;\mu_2,\Sigma_2).
$$

We want to characterize

$$
p_1(x)p_2(x).
$$

Ignoring terms independent of $x$,

$$
\begin{aligned}
\log[p_1(x)p_2(x)]
={}&-\frac12(x-\mu_1)^\top
\Sigma_1^{-1}(x-\mu_1)\\
&-\frac12(x-\mu_2)^\top
\Sigma_2^{-1}(x-\mu_2)
+\text{constant}.
\end{aligned}
$$

Expand the quadratic terms:

$$
\begin{aligned}
\log[p_1(x)p_2(x)]
={}&-\frac12 x^\top
(\Sigma_1^{-1}+\Sigma_2^{-1})x\\
&+x^\top
(\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2)
+\text{constant}.
\end{aligned}
$$

Define the precision matrix

$$
\Lambda=\Sigma^{-1}.
$$

Matching this expression to the log-density of a Gaussian gives

$$
\boxed{
\Sigma
=
(\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
}
$$

and

$$
\boxed{
\mu
=
\Sigma(
\Sigma_1^{-1}\mu_1+
\Sigma_2^{-1}\mu_2
).
}
$$

Therefore,

$$
\boxed{
\mathcal N(x;\mu_1,\Sigma_1)
\mathcal N(x;\mu_2,\Sigma_2)
\propto
\mathcal N(x;\mu,\Sigma).
}
$$

The proportionality symbol is important: the product of two normalized densities is not generally normalized.

### 5.2 Scalar Case

Suppose

$$
p_1(x)=\mathcal N(x;\mu_1,\sigma_1^2)
$$

and

$$
p_2(x)=\mathcal N(x;\mu_2,\sigma_2^2).
$$

Then

$$
\boxed{
\sigma^2
=
\left(
\frac1{\sigma_1^2}
+
\frac1{\sigma_2^2}
\right)^{-1}
}
$$

and

$$
\boxed{
\mu
=
\sigma^2
\left(
\frac{\mu_1}{\sigma_1^2}
+
\frac{\mu_2}{\sigma_2^2}
\right).
}
$$

Equivalently,

$$
\mu
=
\frac{
\sigma_2^2\mu_1+\sigma_1^2\mu_2
}{
\sigma_1^2+\sigma_2^2
}.
$$

The combined mean gives greater weight to the more precise distribution.

Since precision is inverse variance, a small variance corresponds to a large weight.

### 5.3 Connection to Bayesian Inference

Suppose our prior is

$$
X\sim\mathcal N(m_0,P_0)
$$

and an observation satisfies

$$
Y\mid X=x\sim\mathcal N(x,R).
$$

By Bayes' rule,

$$
p(x\mid y)
\propto
p(y\mid x)p(x).
$$

Multiplying the two Gaussian factors produces

$$
P_{\text{post}}
=
(P_0^{-1}+R^{-1})^{-1}
$$

and

$$
m_{\text{post}}
=
P_{\text{post}}
(P_0^{-1}m_0+R^{-1}y).
$$

This is Gaussian Bayesian updating.

The same underlying principle appears in Kalman filtering and the exact Gaussian posterior used in diffusion models.

---

## 6. KL Divergence Between Gaussian Distributions

We derived the general idea of KL divergence in M01.

Here we revisit Gaussian KL divergence to understand its matrix structure.

### 6.1 General Formula

Let

$$
p=\mathcal N(\mu_p,\Sigma_p)
$$

and

$$
q=\mathcal N(\mu_q,\Sigma_q),
$$

where both covariance matrices are positive definite.

Then

$$
\boxed{
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
=\frac12\Big[
&\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)\\
&+(\mu_q-\mu_p)^\top
\Sigma_q^{-1}(\mu_q-\mu_p)\\
&-d+
\log\frac{|\Sigma_q|}{|\Sigma_p|}
\Big].
\end{aligned}
}
$$

### 6.2 Derivation

By definition,

$$
D_{\mathrm{KL}}(p\|q)
=
\mathbb E_p[
\log p(X)-\log q(X)
].
$$

Substituting the Gaussian log-densities,

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
=\frac12\Big[
&\log\frac{|\Sigma_q|}{|\Sigma_p|}\\
&+\mathbb E_p[
(X-\mu_q)^\top
\Sigma_q^{-1}(X-\mu_q)
]\\
&-\mathbb E_p[
(X-\mu_p)^\top
\Sigma_p^{-1}(X-\mu_p)
]
\Big].
\end{aligned}
$$

We use the identity

$$
\boxed{
\mathbb E[
(X-a)^\top A(X-a)
]
=
\operatorname{tr}(A\Sigma)
+
(\mu-a)^\top A(\mu-a).
}
$$

Applying it to the first quadratic expectation gives

$$
\begin{aligned}
&\mathbb E_p[
(X-\mu_q)^\top
\Sigma_q^{-1}(X-\mu_q)
]\\
&=
\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)
+
(\mu_p-\mu_q)^\top
\Sigma_q^{-1}(\mu_p-\mu_q).
\end{aligned}
$$

For the second expectation,

$$
\begin{aligned}
&\mathbb E_p[
(X-\mu_p)^\top
\Sigma_p^{-1}(X-\mu_p)
]\\
&=\operatorname{tr}(\Sigma_p^{-1}\Sigma_p)\\
&=d.
\end{aligned}
$$

Combining these results gives the Gaussian KL formula.

### 6.3 Isotropic Gaussian KL

Suppose

$$
p=\mathcal N(\mu_p,\sigma_p^2I)
$$

and

$$
q=\mathcal N(\mu_q,\sigma_q^2I).
$$

Then

$$
\boxed{
D_{\mathrm{KL}}(p\|q)
=
\frac12
\left[
d\left(
\frac{\sigma_p^2}{\sigma_q^2}
-1+
\log\frac{\sigma_q^2}{\sigma_p^2}
\right)
+
\frac{\|\mu_p-\mu_q\|^2}{\sigma_q^2}
\right].
}
$$

If the two covariance matrices are identical,

$$
\Sigma_p=\Sigma_q=\sigma^2I,
$$

then

$$
\boxed{
D_{\mathrm{KL}}(p\|q)
=
\frac{
\|\mu_p-\mu_q\|^2
}{
2\sigma^2
}.
}
$$

This special case is important in DDPM.

When two Gaussian distributions have the same fixed variance, minimizing their KL divergence is equivalent to minimizing a weighted squared difference between their means.

This connection helps explain why the DDPM variational objective can be related to an MSE objective.

### 6.4 Diagonal Gaussian vs Standard Normal

For completeness, recall the VAE result:

$$
q(z\mid x)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2)
)
$$

and

$$
p(z)=\mathcal N(0,I).
$$

Then

$$
\boxed{
D_{\mathrm{KL}}(q\|p)
=
\frac12\sum_{j=1}^d
\left(
\mu_j^2+\sigma_j^2
-1-\log\sigma_j^2
\right).
}
$$

Thus, Gaussian KL provides an analytical regularization term in VAEs and will appear again in DDPM.

---

## 7. Matrix Calculus Essentials

We now collect several matrix derivatives frequently used in Gaussian models.

### 7.1 Gradient of a Quadratic Form

Let

$$
f(x)=x^\top Ax.
$$

For a general matrix $A$,

$$
\boxed{
\nabla_x f(x)
=
(A+A^\top)x.
}
$$

If $A$ is symmetric,

$$
\boxed{
\nabla_x(x^\top Ax)=2Ax.
}
$$

For a centered quadratic form,

$$
f(x)=(x-\mu)^\top A(x-\mu),
$$

we obtain

$$
\boxed{
\nabla_x f(x)
=
2A(x-\mu)
}
$$

when $A$ is symmetric.

### 7.2 Gradient of a Gaussian Log-Density

Recall

$$
\log p(x)
=
-\frac12(x-\mu)^\top
\Sigma^{-1}(x-\mu)
+\text{constant},
$$

where the constant does not depend on $x$.

Differentiating,

$$
\boxed{
\nabla_x\log p(x)
=
-\Sigma^{-1}(x-\mu).
}
$$

This quantity is known as the **score function** of the probability density.

For

$$
p(x)=\mathcal N(0,\sigma^2I),
$$

the score is

$$
\boxed{
\nabla_x\log p(x)
=
-\frac{x}{\sigma^2}.
}
$$

The score points toward regions of higher probability density.

This concept becomes central in score-based diffusion models.

### 7.3 Gradient of the Log-Determinant

For an invertible matrix $A$,

$$
d\log|\det A|
=
\operatorname{tr}(A^{-1}dA).
$$

Therefore,

$$
\boxed{
\nabla_A\log|\det A|
=
A^{-\top}.
}
$$

For a symmetric positive-definite matrix,

$$
\nabla_A\log|A|=A^{-1}.
$$

The log-determinant appears in Gaussian likelihoods, Gaussian KL divergence, and normalizing flow models.

### 7.4 Derivative of a Matrix Inverse

Starting from

$$
AA^{-1}=I,
$$

take differentials:

$$
(dA)A^{-1}+A(dA^{-1})=0.
$$

Multiplying from the left by $A^{-1}$,

$$
\boxed{
d(A^{-1})
=
-A^{-1}(dA)A^{-1}.
}
$$

This identity is useful when covariance matrices or precision matrices depend on model parameters.

### 7.5 Useful Trace Identities

For compatible matrices,

$$
\operatorname{tr}(AB)
=
\operatorname{tr}(BA).
$$

More generally, cyclic permutations preserve the trace:

$$
\operatorname{tr}(ABC)
=
\operatorname{tr}(BCA).
$$

For vectors $x,y$,

$$
x^\top Ay
=
\operatorname{tr}(Ayx^\top).
$$

These identities help convert scalar quadratic expressions into forms that are easier to differentiate.

---

## 8. Gaussian Reparameterization

The reparameterization trick was introduced in M03 for variational inference.

Here we state the general Gaussian version.

Suppose

$$
X\sim\mathcal N(\mu,\Sigma).
$$

Let

$$
\Sigma=LL^\top
$$

be a Cholesky factorization.

If

$$
\epsilon\sim\mathcal N(0,I),
$$

then

$$
\boxed{
X=\mu+L\epsilon
}
$$

has the desired Gaussian distribution.

For an isotropic covariance,

$$
\Sigma=\sigma^2I,
$$

this reduces to

$$
\boxed{
X=\mu+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

### 8.1 Connection to VAE

A VAE encoder produces

$$
\mu_\phi(x),\qquad
\sigma_\phi(x).
$$

Sampling from the approximate posterior is implemented as

$$
z=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon.
$$

### 8.2 Connection to DDPM

In DDPM, a noisy sample can be generated through

$$
\boxed{
x_t=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
}
$$

where

$$
\epsilon\sim\mathcal N(0,I).
$$

The algebra resembles VAE reparameterization, but the role is different.

In a VAE, the encoder distribution is learned.

In the DDPM forward process, the noise distribution and variance schedule are specified in advance.

The network is then trained to learn the reverse process.

---

## 9. Numerical Experiments

### Experiment 1: Conditional Gaussian Distribution

We verify that the conditional mean and variance match the theoretical formulas.

For a bivariate standard Gaussian with correlation $\rho$,

$$
X\mid Y=y
\sim
\mathcal N(\rho y,1-\rho^2).
$$

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

rho = 0.8
n = 200_000

mean = np.zeros(2)
cov = np.array([
    [1.0, rho],
    [rho, 1.0]
])

samples = rng.multivariate_normal(mean, cov, size=n)
x = samples[:, 0]
y = samples[:, 1]

# Approximate conditioning on Y = 1
y_target = 1.0
width = 0.05
mask = np.abs(y - y_target) < width

conditional_x = x[mask]

theoretical_mean = rho * y_target
theoretical_var = 1 - rho**2

print("Empirical conditional mean:", conditional_x.mean())
print("Theoretical mean:", theoretical_mean)

print("Empirical conditional variance:", conditional_x.var())
print("Theoretical variance:", theoretical_var)

plt.hist(
    conditional_x,
    bins=45,
    density=True,
    alpha=0.6,
    label="Empirical"
)

grid = np.linspace(-2, 4, 400)

pdf = (
    np.exp(
        -(grid - theoretical_mean)**2
        / (2 * theoretical_var)
    )
    / np.sqrt(2 * np.pi * theoretical_var)
)

plt.plot(grid, pdf, label="Theoretical")
plt.legend()
plt.title("Conditional Gaussian")
plt.show()
```

Note that conditioning on the event $Y=1$ exactly is not possible by filtering continuous random samples because that event has probability zero.

We therefore approximate the conditional distribution using a narrow interval around $Y=1$.

As the interval becomes smaller, the approximation becomes more accurate, provided enough samples remain.

### Experiment 2: Gaussian KL vs Monte Carlo

We compare the analytical KL divergence with a Monte Carlo estimate.

```python
import numpy as np

rng = np.random.default_rng(42)

mu_p = np.array([1.0, -0.5])
mu_q = np.array([0.0, 0.0])

sigma_p = np.array([0.5, 1.5])
sigma_q = np.array([1.0, 1.0])

var_p = sigma_p**2
var_q = sigma_q**2

# Analytical KL
kl_analytic = 0.5 * np.sum(
    var_p / var_q
    + (mu_p - mu_q)**2 / var_q
    - 1
    + np.log(var_q / var_p)
)

# Monte Carlo KL
n = 100_000
samples = (
    mu_p
    + rng.normal(size=(n, 2)) * sigma_p
)

def diagonal_gaussian_logpdf(x, mu, var):
    return -0.5 * np.sum(
        np.log(2 * np.pi * var)
        + (x - mu)**2 / var,
        axis=-1
    )

log_p = diagonal_gaussian_logpdf(
    samples, mu_p, var_p
)

log_q = diagonal_gaussian_logpdf(
    samples, mu_q, var_q
)

kl_mc = np.mean(log_p - log_q)

print("Analytical KL:", kl_analytic)
print("Monte Carlo KL:", kl_mc)
```

The Monte Carlo estimate should approach the analytical value as the number of samples increases.

### Experiment 3: Verify Gaussian Addition

```python
import numpy as np

rng = np.random.default_rng(42)

n = 100_000
alpha = 0.7
beta = 0.3

x = rng.normal(
    loc=2.0,
    scale=1.5,
    size=n
)

eps = rng.normal(size=n)

y = np.sqrt(alpha) * x + np.sqrt(beta) * eps

expected_mean = np.sqrt(alpha) * 2.0
expected_var = alpha * (1.5**2) + beta

print("Empirical mean:", y.mean())
print("Expected mean:", expected_mean)

print("Empirical variance:", y.var())
print("Expected variance:", expected_var)
```

This experiment demonstrates the Gaussian propagation identity that will be used repeatedly in DDPM.

---

## 10. Common Pitfalls

### 10.1 Covariance vs Standard Deviation

If

$$
X\sim\mathcal N(\mu,\sigma^2I),
$$

the covariance is $\sigma^2I$, but the noise scaling in a reparameterization is $\sigma$.

Therefore,

$$
X=\mu+\sigma\epsilon,
$$

not

$$
X=\mu+\sigma^2\epsilon.
$$

### 10.2 Correlation Does Not Always Imply Gaussianity

A vector may have Gaussian marginal distributions without being jointly Gaussian.

Therefore, the conditional Gaussian formula requires a joint Gaussian assumption, not merely Gaussian marginals.

### 10.3 Gaussian Products Require a Common Variable

The product identity applies to two density factors viewed as functions of the same variable.

It should not be confused with multiplying independent Gaussian random variables.

The random variable $XY$ is not generally Gaussian even when $X$ and $Y$ are Gaussian.

### 10.4 KL Divergence Is Not Symmetric

In general,

$$
D_{\mathrm{KL}}(p\|q)
\neq
D_{\mathrm{KL}}(q\|p).
$$

The covariance of the second argument appears inside the inverse matrix of the general Gaussian KL formula.

### 10.5 Avoid Explicit Matrix Inversion in Code

Expressions such as

$$
\Sigma^{-1}b
$$

should normally be computed with a linear solver.

For example:

```python
# Prefer:
x = np.linalg.solve(Sigma, b)

# Instead of:
x = np.linalg.inv(Sigma) @ b
```

For log-determinants, `np.linalg.slogdet` is often more numerically stable than taking the logarithm of a directly computed determinant.

---

## 11. Summary and Connection to DDPM

The main Gaussian identities are:

| Concept                         | Main result                                          |
| ------------------------------- | ---------------------------------------------------- |
| Affine Gaussian                 | $AX+b\sim\mathcal N(A\mu+b,A\Sigma A^\top)$          |
| Independent Gaussian addition   | Covariances add                                      |
| Conditional Gaussian mean       | $\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)$         |
| Conditional Gaussian covariance | $\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$ |
| Gaussian product covariance     | $(\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}$                 |
| Gaussian score                  | $-\Sigma^{-1}(x-\mu)$                                |
| Gaussian reparameterization     | $X=\mu+L\epsilon$                                    |

The most important connections to diffusion models are:

**First**, Gaussian addition allows us to derive the distribution of a noisy sample directly from its original clean sample.

**Second**, Gaussian conditioning and Gaussian products allow us to derive the exact forward-process posterior used in DDPM training.

**Third**, Gaussian KL divergence explains how differences between Gaussian reverse-transition means lead to quadratic loss functions.

**Fourth**, the gradient of a Gaussian log-density introduces the score function, which becomes important in continuous-time score-based diffusion models.

The next mathematical module introduces discrete-time stochastic processes and Markov chains.

Together, M04 and M05 provide the mathematical foundation needed to derive the DDPM forward process and its variational objective.

---

## Exercises

1. Let $X\sim\mathcal N(\mu,\Sigma)$ and $Y=2X+3b$. Derive the mean and covariance of $Y$.

2. Suppose $(X,Y)$ is a bivariate standard Gaussian with correlation $\rho=0.6$. Derive $X\mid Y=2$.

3. Multiply the densities $\mathcal N(x;0,1)$ and $\mathcal N(x;2,4)$. Find the mean and variance of the normalized product.

4. Show that, when $\Sigma_p=\Sigma_q=\sigma^2I$,

   $$
   D_{\mathrm{KL}}(p\|q)
   =
   \frac{\|\mu_p-\mu_q\|^2}{2\sigma^2}.
   $$

5. For $p(x)=\mathcal N(0,\sigma^2I)$, derive $\nabla_x\log p(x)$.

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 2.

2. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.

3. Petersen, K. B., & Pedersen, M. S. (2012). *The Matrix Cookbook*.

4. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.

5. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.