---
layout: post
title: "M03 - Variational Inference"
date: 2026-10-15
mathjax: true
comments: true
---

# Variational Inference

**Prerequisites:** M01 — Probability & Information Theory; M02 — Statistical Inference & Latent Variable Models.

**Aim:** Understand how variational inference transforms intractable posterior inference into optimization, derive mean-field coordinate updates, and develop the mathematical foundations of amortized inference and variational autoencoders.

## 1. Motivation: Why Variational Inference?

Consider a latent-variable generative model

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z),
$$

where $x$ denotes observed data, $z$ represents latent variables, and $\theta$ denotes model parameters.

After observing $x$, we may wish to infer the latent variable through its posterior distribution:

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

The marginal likelihood, also called the evidence, is

$$
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
$$

Although evaluating the joint density $p_\theta(x,z)$ may be straightforward, evaluating this integral can be computationally expensive or analytically intractable.

For example, when $z$ is high-dimensional and the decoder $p_\theta(x\mid z)$ is represented by a nonlinear neural network, the posterior generally does not belong to a convenient parametric family.

One possible approach is to approximate posterior expectations using Monte Carlo methods, such as MCMC.

Variational inference takes a different approach.

Instead of attempting to compute the true posterior directly, we introduce a family of tractable distributions

$$
\mathcal Q_x
=
\{q_\phi(z\mid x):\phi\in\Phi\}
$$

and search for the member closest to the true posterior.

The optimization problem is

$$
\boxed{
\phi^*
=
\operatorname*{arg\,min}_{\phi}
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
}
$$

Here, the divergence is the **reverse KL divergence**, with the approximate distribution appearing as the first argument.

The key idea is that posterior inference becomes an optimization problem over probability distributions.

## 2. The Variational Objective

### 2.1 From KL Divergence to ELBO

Recall the identity derived in M02:

$$
\boxed{
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
}
$$

The Evidence Lower Bound is

$$
\boxed{
\mathcal L(\theta,\phi;x)
=
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
}
$$

Equivalently,

$$
\begin{aligned}
\mathcal L(\theta,\phi;x)
={}&
\mathbb E_{z\sim q_\phi(z\mid x)}
[\log p_\theta(x,z)]\\
&-
\mathbb E_{z\sim q_\phi(z\mid x)}
[\log q_\phi(z\mid x)].
\end{aligned}
$$

Since KL divergence is nonnegative,

$$
\mathcal L(\theta,\phi;x)
\leq
\log p_\theta(x).
$$

For a fixed model parameter $\theta$, the marginal log-likelihood is constant with respect to $\phi$.

Therefore,

$$
\begin{aligned}
&\operatorname*{arg\,min}_{\phi}
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right)\\
&=
\operatorname*{arg\,max}_{\phi}
\mathcal L(\theta,\phi;x).
\end{aligned}
$$

Thus,

$$
\boxed{
\phi^*
=
\operatorname*{arg\,max}_{\phi}
\mathcal L(\theta,\phi;x).
}
$$

This is the fundamental optimization principle of variational inference.

Importantly, evaluating the ELBO only requires access to the joint density $p_\theta(x,z)$ and the variational distribution $q_\phi(z\mid x)$, together with the ability to evaluate or approximate the relevant expectations.

We do not need to evaluate the intractable normalizing constant $p_\theta(x)$.

### 2.2 Reconstruction and Regularization

Suppose the joint distribution factorizes as

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z).
$$

Expanding the ELBO gives

$$
\begin{aligned}
\mathcal L(\theta,\phi;x)
&=
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]\\
&=
\mathbb E_{z\sim q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]\\
&\quad+
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log\frac{p(z)}{q_\phi(z\mid x)}
\right].
\end{aligned}
$$

Therefore,

$$
\boxed{
\begin{aligned}
\mathcal L(\theta,\phi;x)
={}&
\underbrace{
\mathbb E_{z\sim q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
}_{\text{Expected reconstruction log-likelihood}}\\
&-
\underbrace{
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p(z)\right)
}_{\text{Prior regularization}}.
\end{aligned}
}
$$

The two terms serve different purposes.

The expected log-likelihood encourages latent variables that explain the observed data.

The KL term penalizes the approximate posterior for deviating from the prior distribution.

However, this KL divergence should not be confused with the approximation gap

$$
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
$$

The latter measures how far the ELBO lies below the true log evidence, whereas the former is part of the ELBO objective itself.

## 3. Forward KL and Reverse KL

KL divergence is asymmetric:

$$
D_{\mathrm{KL}}(p\|q)
\neq
D_{\mathrm{KL}}(q\|p).
$$

Suppose $p_\theta(z\mid x)$ is the true posterior distribution and $q_\phi(z\mid x)$ is its approximation for a fixed observation $x$.

### 3.1 Forward KL

Forward KL is defined as

$$
\boxed{
D_{\mathrm{KL}}\left(p_\theta(z\mid x)\|q_\phi(z\mid x)\right)
=
\mathbb E_{z\sim p_\theta(z\mid x)}
\left[
\log\frac{p_\theta(z\mid x)}{q_\phi(z\mid x)}
\right].
}
$$

Forward KL heavily penalizes assigning low probability density under the approximation to regions where the true posterior assigns substantial probability mass.

For restricted approximation families, this can encourage **mode-covering** behavior: the approximate distribution may spread its probability mass across multiple modes of the target distribution.

### 3.2 Reverse KL

Reverse KL is defined as

$$
\boxed{
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right)
=
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right].
}
$$

Reverse KL penalizes placing probability mass in regions where the true posterior density is very small.

For a unimodal approximation of a multimodal target, this can encourage **mode-seeking** behavior: the approximate distribution may concentrate around one mode rather than covering all modes.

These behaviors are tendencies rather than universal rules. They depend on both the target distribution and the chosen approximation family.

### 3.3 Why Does Variational Inference Use Reverse KL?

Standard ELBO-based variational inference minimizes the reverse KL divergence between the approximate posterior and the true posterior:

$$
\boxed{
\min_\phi
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
}
$$

However, directly evaluating this divergence is generally difficult because the true posterior contains the intractable marginal likelihood $p_\theta(x)$.

Recall the KL decomposition:

$$
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right).
$$

For fixed $\theta$ and $x$, the marginal log-likelihood $\log p_\theta(x)$ is constant with respect to $\phi$.

Therefore,

$$
\boxed{
\min_\phi
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\right)
\iff
\max_\phi\mathcal L(\theta,\phi;x).
}
$$

The ELBO can be evaluated or estimated using the joint density $p_\theta(x,z)$ and the approximate posterior $q_\phi(z\mid x)$, without explicitly computing the marginal likelihood $p_\theta(x)$.

This makes reverse-KL optimization computationally practical even when the exact posterior is intractable.

In contrast, forward KL involves an expectation under the true posterior $p_\theta(z\mid x)$, which may itself be difficult to sample from or integrate over.

Thus, the reverse-KL formulation provides a convenient optimization objective for standard variational inference.

## 4. Mean-Field Variational Inference

### 4.1 The Mean-Field Assumption

Suppose the latent variable consists of $m$ components:

$$
z=(z_1,z_2,\ldots,z_m).
$$

Our goal is to approximate the true posterior distribution

$$
p_\theta(z_1,\ldots,z_m\mid x).
$$

However, the latent variables may exhibit complicated dependencies, making the joint posterior difficult to represent and optimize.

**Mean-field variational inference** introduces a simplifying assumption: the components of the approximate posterior are conditionally independent given the observation $x$.

Specifically, we assume

$$
\boxed{
q(z_1,\ldots,z_m\mid x)
=
\prod_{j=1}^{m}q_j(z_j\mid x).
}
$$

Importantly, this factorization is imposed on the **approximate posterior**, not on the true posterior.

In general,

$$
p_\theta(z\mid x)
\neq
\prod_{j=1}^{m}p_\theta(z_j\mid x).
$$

The true posterior may contain dependencies that the mean-field approximation cannot represent.

For the following derivation, we consider a fixed observation $x$ and fixed model parameters $\theta$.

To simplify notation, we suppress the conditioning on $x$ and write

$$
\boxed{
q(z)=\prod_{j=1}^{m}q_j(z_j).
}
$$

The advantage of this factorization is that instead of optimizing an unrestricted joint distribution, we can optimize each variational factor separately while holding the others fixed.

This leads to **Coordinate Ascent Variational Inference (CAVI)**.

### 4.2 Deriving the Coordinate Update

Recall that the ELBO is

$$
\boxed{
\mathcal L(q)
=
\mathbb E_{z\sim q(z)}
\left[
\log\frac{p_\theta(x,z)}{q(z)}
\right].
}
$$

Equivalently,

$$
\mathcal L(q)
=
\mathbb E_q[\log p_\theta(x,z)]
-
\mathbb E_q[\log q(z)].
$$

We want to maximize the ELBO over the mean-field variational family:

$$
\max_{q_1,\ldots,q_m}\mathcal L(q).
$$

Rather than optimizing every factor simultaneously, consider updating one particular factor $q_j(z_j)$ while holding all the remaining factors fixed.

Define

$$
q_{-j}(z_{-j})
=
\prod_{k\neq j}q_k(z_k),
$$

where $z_{-j}$ denotes all latent variables except $z_j$.

Then

$$
q(z)=q_j(z_j)q_{-j}(z_{-j}).
$$

**Step 1: Expand the expected joint log-likelihood**

The first term of the ELBO is

$$
\mathbb E_{z\sim q(z)}
[\log p_\theta(x,z)].
$$

Using the mean-field factorization,

$$
\begin{aligned}
&\mathbb E_{z\sim q(z)}
[\log p_\theta(x,z)]\\
&=
\int\int
q_j(z_j)q_{-j}(z_{-j})
\log p_\theta(x,z)
\,dz_{-j}\,dz_j\\
&=
\int q_j(z_j)
\underbrace{
\mathbb E_{z_{-j}\sim q_{-j}}
[\log p_\theta(x,z)]
}_{A_j(z_j)}
\,dz_j.
\end{aligned}
$$

Define

$$
\boxed{
A_j(z_j)
=
\mathbb E_{z_{-j}\sim q_{-j}}
[\log p_\theta(x,z)].
}
$$

The expectation integrates out all latent variables except $z_j$.

Consequently, $A_j(z_j)$ is a function of $z_j$ alone.

**Step 2: Expand the entropy term**

The second term is

$$
-\mathbb E_{z\sim q(z)}[\log q(z)].
$$

Since

$$
q(z)=\prod_{k=1}^{m}q_k(z_k),
$$

taking logarithms gives

$$
\log q(z)
=
\sum_{k=1}^{m}\log q_k(z_k).
$$

Separating the $j$-th factor,

$$
\log q(z)
=
\log q_j(z_j)
+
\sum_{k\neq j}\log q_k(z_k).
$$

Therefore,

$$
\begin{aligned}
-\mathbb E_q[\log q(z)]
={}&
-\int q_j(z_j)\log q_j(z_j)\,dz_j\\
&-
\sum_{k\neq j}
\mathbb E_{z_k\sim q_k}[\log q_k(z_k)].
\end{aligned}
$$

Because all factors $q_k$ for $k\neq j$ are fixed, the second term is constant with respect to $q_j$.

**Step 3: Construct the optimization objective**

Combining the two terms, we obtain

$$
\boxed{
\begin{aligned}
\mathcal L(q_j)
={}&
\int q_j(z_j)A_j(z_j)\,dz_j\\
&-
\int q_j(z_j)\log q_j(z_j)\,dz_j
+C,
\end{aligned}
}
$$

where $C$ contains all terms independent of $q_j$.

The optimization is subject to the probability normalization constraint

$$
\int q_j(z_j)\,dz_j=1.
$$

**Step 4: Apply a Lagrange multiplier**

Introduce a Lagrange multiplier $\lambda$:

$$
\begin{aligned}
\mathcal J(q_j,\lambda)
={}&
\int q_j(z_j)A_j(z_j)\,dz_j\\
&-
\int q_j(z_j)\log q_j(z_j)\,dz_j\\
&+
\lambda\left(
\int q_j(z_j)\,dz_j-1
\right).
\end{aligned}
$$

Taking the functional derivative with respect to $q_j(z_j)$ gives

$$
\frac{\delta\mathcal J}{\delta q_j(z_j)}
=
A_j(z_j)-\log q_j(z_j)-1+\lambda.
$$

Setting this derivative equal to zero,

$$
A_j(z_j)-\log q_j^*(z_j)-1+\lambda=0.
$$

Rearranging,

$$
\log q_j^*(z_j)
=
A_j(z_j)+\lambda-1.
$$

Exponentiating both sides,

$$
q_j^*(z_j)
=
e^{\lambda-1}\exp(A_j(z_j)).
$$

Since $e^{\lambda-1}$ does not depend on $z_j$, it acts as a normalization constant.

Thus,

$$
q_j^*(z_j)
\propto
\exp(A_j(z_j)).
$$

Substituting the definition of $A_j(z_j)$ gives the fundamental CAVI update:

$$
\boxed{
q_j^*(z_j)
\propto
\exp\left(
\mathbb E_{z_{-j}\sim q_{-j}}
[\log p_\theta(x,z)]
\right).
}
$$

More explicitly, the normalized solution is

$$
\boxed{
q_j^*(z_j)
=
\frac{
\exp\left(
\mathbb E_{z_{-j}\sim q_{-j}}
[\log p_\theta(x,z)]
\right)
}{
\displaystyle
\int
\exp\left(
\mathbb E_{z_{-j}\sim q_{-j}}
[\log p_\theta(x,z)]
\right)
\,dz_j
}.
}
$$

The denominator ensures that $q_j^*$ integrates to one.

This update assumes that the normalizing integral exists and that the relevant expectations are well-defined.

### 4.3 Why Is This Coordinate Ascent?

The mean-field variational objective is

$$
\max_{q_1,\ldots,q_m}
\mathcal L(q_1,\ldots,q_m).
$$

CAVI optimizes one variational factor at a time.

At iteration $t$, the update for factor $j$ can be expressed as

$$
\boxed{
q_j^{(t+1)}
=
\operatorname*{arg\,max}_{q_j}
\mathcal L
\left(
q_1^{(t+1)},\ldots,
q_{j-1}^{(t+1)},
q_j,
q_{j+1}^{(t)},\ldots,q_m^{(t)}
\right).
}
$$

This expression corresponds to sequential coordinate updates, where previously updated factors use their new values and the remaining factors retain their previous values.

The algorithm repeatedly cycles through the factors:

$$
q_1\rightarrow q_2\rightarrow\cdots
\rightarrow q_m\rightarrow q_1\rightarrow\cdots
$$

Each exact coordinate update maximizes the ELBO with respect to one factor while holding the others fixed.

Therefore, the ELBO is non-decreasing:

$$
\boxed{
\mathcal L(q^{(t+1)})
\geq
\mathcal L(q^{(t)}).
}
$$

However, monotonic improvement does not guarantee convergence to the global optimum.

The mean-field factorization restricts the family of distributions that can be represented, and the joint optimization over all factors may have multiple stationary points.

Consequently, CAVI can converge to a locally optimal approximation that differs from the true posterior.

### 4.4 Connection to Variational EM

CAVI and EM both use coordinate-ascent optimization, but they optimize different objects.

In **Exact EM**, the E-step selects the exact posterior, while the M-step updates the model parameters:

$$
q^{(t+1)}(z)
=
p_{\theta^{(t)}}(z\mid x),
$$

$$
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
\mathcal L(q^{(t+1)},\theta).
$$

In **Mean-field CAVI**, the model parameters $\theta$ are fixed, and the approximate posterior is optimized by updating its individual factors.

In **Variational EM**, these two procedures can be combined:

1. Use CAVI or another variational optimization method to approximately infer the latent variables.
2. Update the model parameters using the resulting approximate posterior.

Thus, mean-field CAVI provides a practical approach for performing approximate E-steps when exact posterior inference is computationally intractable.

## 5. Example: Mean-Field Approximation of a Gaussian

Consider a latent variable represented by a two-dimensional random vector

$$
\mathbf z=
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix}
\in\mathbb R^2.
$$

Here, $z_1$ and $z_2$ are scalar random variables, and $\mathbf z$ denotes the complete latent vector.

Given a fixed observation $x$, suppose the true posterior distribution is a correlated bivariate Gaussian:

$$
\boxed{
p_\theta(\mathbf z\mid x)
=
\mathcal N_2
\left(
\mathbf z;
\underbrace{
\begin{bmatrix}
0\\
0
\end{bmatrix}
}_{\boldsymbol\mu},
\underbrace{
\begin{bmatrix}
1&\rho\\
\rho&1
\end{bmatrix}
}_{\boldsymbol\Sigma}
\right),
\qquad |\rho|<1.
}
$$

The notation $\mathcal N_2(\mathbf z;\boldsymbol\mu,\boldsymbol\Sigma)$ denotes the probability density of a two-dimensional Gaussian distribution, evaluated at $\mathbf z$.

The mean vector is

$$
\boldsymbol\mu=
\begin{bmatrix}
0\\0
\end{bmatrix},
$$

and the covariance matrix is

$$
\boldsymbol\Sigma=
\begin{bmatrix}
1&\rho\\
\rho&1
\end{bmatrix}.
$$

Therefore,

$$
\operatorname{Var}(Z_1\mid x)
=
\operatorname{Var}(Z_2\mid x)=1,
$$

and

$$
\operatorname{Cov}(Z_1,Z_2\mid x)=\rho.
$$

Because both marginal variances equal one, $\rho$ is also the posterior correlation coefficient.

When $\rho\neq0$, the latent variables are correlated under the true posterior.

### 5.1 Constructing the Mean-Field Approximation

We wish to approximate the true posterior $p_\theta(\mathbf z\mid x)$ using a factorized variational distribution:

$$
\boxed{
q(\mathbf z\mid x)
=
q_1(z_1\mid x)q_2(z_2\mid x).
}
$$

Here, $q_1(z_1\mid x)$ and $q_2(z_2\mid x)$ are one-dimensional probability densities.

The mean-field assumption requires $Z_1$ and $Z_2$ to be conditionally independent under the approximate posterior $q$.

This is generally inconsistent with the true posterior when $\rho\neq0$.

Our objective is to find

$$
\boxed{
q^*
=
\operatorname*{arg\,min}_{q\in\mathcal Q_{\mathrm{MF}}}
D_{\mathrm{KL}}
\left(
q(\mathbf z\mid x)\|
p_\theta(\mathbf z\mid x)
\right).
}
$$

Equivalently, we maximize the ELBO within the mean-field family $\mathcal Q_{\mathrm{MF}}$.

For this example, the observation $x$ and model parameters $\theta$ are fixed throughout the optimization.

### 5.2 Expanding the True Posterior Log-Density

Recall that the density of a two-dimensional Gaussian is

$$
p_\theta(\mathbf z\mid x)
=
\frac{1}
{2\pi\sqrt{\det\boldsymbol\Sigma}}
\exp
\left[
-\frac12
(\mathbf z-\boldsymbol\mu)^\top
\boldsymbol\Sigma^{-1}
(\mathbf z-\boldsymbol\mu)
\right].
$$

Since

$$
\det\boldsymbol\Sigma=1-\rho^2,
$$

the inverse covariance matrix is

$$
\boldsymbol\Sigma^{-1}
=
\frac{1}{1-\rho^2}
\begin{bmatrix}
1&-\rho\\
-\rho&1
\end{bmatrix}.
$$

Therefore, the quadratic form becomes

$$
\mathbf z^\top\boldsymbol\Sigma^{-1}\mathbf z
=
\frac{
z_1^2-2\rho z_1z_2+z_2^2
}{
1-\rho^2
}.
$$

Taking logarithms gives

$$
\boxed{
\log p_\theta(\mathbf z\mid x)
=
-\frac{
z_1^2-2\rho z_1z_2+z_2^2
}{
2(1-\rho^2)
}
+C,
}
$$

where $C$ contains terms independent of $z_1$ and $z_2$.

### 5.3 Updating the First Variational Factor

From Section 4, the CAVI update for the first factor is

$$
\boxed{
\log q_1^*(z_1\mid x)
=
\mathbb E_{z_2\sim q_2(\cdot\mid x)}
\left[
\log p_\theta(\mathbf z\mid x)
\right]
+C_1.
}
$$

The expectation is taken over $z_2$, while $z_1$ remains a free variable.

Define the mean of the current second variational factor as

$$
m_2
=
\mathbb E_{z_2\sim q_2(\cdot\mid x)}[z_2].
$$

Substituting the Gaussian log-density,

$$
\begin{aligned}
\log q_1^*(z_1\mid x)
={}&
-\frac{1}{2(1-\rho^2)}
\mathbb E_{z_2\sim q_2(\cdot\mid x)}
\left[
z_1^2-2\rho z_1z_2+z_2^2
\right]
+C_1.
\end{aligned}
$$

Since $z_1$ is fixed with respect to this expectation,

$$
\mathbb E_{q_2}[z_1^2]=z_1^2,
$$

and

$$
\mathbb E_{q_2}[z_1z_2]=z_1m_2.
$$

The term $\mathbb E_{q_2}[z_2^2]$ does not depend on $z_1$, so it can be absorbed into the constant.

Hence,

$$
\boxed{
\log q_1^*(z_1\mid x)
=
-\frac{
z_1^2-2\rho m_2z_1
}{
2(1-\rho^2)
}
+C_1.
}
$$

Completing the square,

$$
z_1^2-2\rho m_2z_1
=
(z_1-\rho m_2)^2-\rho^2m_2^2.
$$

The final term is constant with respect to $z_1$.

Therefore,

$$
\log q_1^*(z_1\mid x)
=
-\frac{
(z_1-\rho m_2)^2
}{
2(1-\rho^2)
}
+C_1'.
$$

Recognizing the Gaussian log-density, we obtain

$$
\boxed{
q_1^*(z_1\mid x)
=
\mathcal N
\left(
z_1;\rho m_2,1-\rho^2
\right).
}
$$

Thus, the coordinate-optimal first factor is Gaussian with mean $\rho m_2$ and variance $1-\rho^2$.

### 5.4 Updating the Second Variational Factor

Similarly, define

$$
m_1
=
\mathbb E_{z_1\sim q_1(\cdot\mid x)}[z_1].
$$

The coordinate update for the second factor is

$$
\log q_2^*(z_2\mid x)
=
\mathbb E_{z_1\sim q_1(\cdot\mid x)}
[\log p_\theta(\mathbf z\mid x)]
+C_2.
$$

Following the same derivation,

$$
\boxed{
q_2^*(z_2\mid x)
=
\mathcal N
\left(
z_2;\rho m_1,1-\rho^2
\right).
}
$$

The two factors are updated alternately until the variational distribution converges.

### 5.5 Solving the Fixed-Point Equations

At a fixed point of CAVI, neither factor changes after a coordinate update.

Therefore, their means must satisfy

$$
\boxed{
m_1=\rho m_2,
\qquad
m_2=\rho m_1.
}
$$

Substituting the second equation into the first,

$$
m_1=\rho^2m_1.
$$

Hence,

$$
(1-\rho^2)m_1=0.
$$

Since $|\rho|<1$, we have $1-\rho^2>0$.

Therefore,

$$
\boxed{
m_1=m_2=0.
}
$$

Substituting these means back into the coordinate-optimal factors gives

$$
q_1^*(z_1\mid x)
=
\mathcal N(z_1;0,1-\rho^2),
$$

and

$$
q_2^*(z_2\mid x)
=
\mathcal N(z_2;0,1-\rho^2).
$$

Consequently, the mean-field approximation is

$$
\boxed{
q^*(\mathbf z\mid x)
=
\mathcal N(z_1;0,1-\rho^2)
\mathcal N(z_2;0,1-\rho^2).
}
$$

Equivalently, this joint distribution is a bivariate Gaussian with diagonal covariance:

$$
\boxed{
q^*(\mathbf z\mid x)
=
\mathcal N_2
\left(
\mathbf z;
\begin{bmatrix}
0\\0
\end{bmatrix},
\begin{bmatrix}
1-\rho^2&0\\
0&1-\rho^2
\end{bmatrix}
\right).
}
$$

### 5.6 Interpretation: Underestimating Posterior Uncertainty

Recall that the true posterior is

$$
p_\theta(\mathbf z\mid x)
=
\mathcal N_2
\left(
\mathbf z;
\begin{bmatrix}
0\\0
\end{bmatrix},
\begin{bmatrix}
1&\rho\\
\rho&1
\end{bmatrix}
\right).
$$

The true posterior marginal variances are

$$
\operatorname{Var}_{p_\theta}(Z_1\mid x)
=
\operatorname{Var}_{p_\theta}(Z_2\mid x)
=
1.
$$

In contrast, the mean-field approximation has marginal variances

$$
\operatorname{Var}_{q^*}(Z_1\mid x)
=
\operatorname{Var}_{q^*}(Z_2\mid x)
=
1-\rho^2.
$$

For example, if

$$
\rho=0.9,
$$

then

$$
1-\rho^2=1-0.81=0.19.
$$

Thus, the approximate marginal variances are only $0.19$, compared with the true marginal variances of $1$.

The approximate posterior also has zero covariance between its two components, whereas the true posterior has covariance $\rho=0.9$.

The mean-field approximation is therefore substantially more concentrated and cannot represent the dependence present in the true posterior.

This illustrates an important limitation of mean-field variational inference:

**When posterior variables are strongly correlated, a factorized approximation optimized using reverse KL can underestimate posterior uncertainty.**

## 6. Amortized Variational Inference

### 6.1 The Computational Problem with Local Inference

Suppose we have a dataset containing $n$ observations:

$$
X=\{x_1,x_2,\ldots,x_n\}.
$$

Assume each observation $x_i$ is associated with its own latent random vector

$$
\mathbf z_i\in\mathbb R^d.
$$

Here, the index $i$ identifies an observation, while the dimension $d$ specifies the number of latent components within each vector.

Under a model with independent observations and sample-specific latent variables, the joint distribution factorizes as

$$
p_\theta(X,Z)
=
\prod_{i=1}^{n}
p(\mathbf z_i)p_\theta(x_i\mid\mathbf z_i),
$$

where $Z=\{\mathbf z_1,\ldots,\mathbf z_n\}$.

In traditional variational inference, we may introduce a separate approximate posterior for each observation:

$$
\boxed{
q_{\phi_i}(\mathbf z_i\mid x_i).
}
$$

Each $\phi_i$ denotes the variational parameters associated with observation $x_i$.

For example, if the variational family consists of diagonal Gaussian distributions,

$$
q_{\phi_i}(\mathbf z_i\mid x_i)
=
\mathcal N_d
\left(
\mathbf z_i;
\boldsymbol\mu_i,
\operatorname{diag}(\boldsymbol\sigma_i^2)
\right),
$$

where

$$
\phi_i=(\boldsymbol\mu_i,\boldsymbol\sigma_i^2).
$$

These variational parameters are optimized independently for different observations, although the underlying generative model parameters $\theta$ are shared.

The objective becomes

$$
\boxed{
\max_{\theta,\phi_1,\ldots,\phi_n}
\sum_{i=1}^{n}
\mathcal L(\theta,\phi_i;x_i).
}
$$

Here,

$$
\mathcal L(\theta,\phi_i;x_i)
=
\mathbb E_{\mathbf z_i\sim q_{\phi_i}(\mathbf z_i\mid x_i)}
\left[
\log
\frac{p_\theta(x_i,\mathbf z_i)}
{q_{\phi_i}(\mathbf z_i\mid x_i)}
\right].
$$

This approach has two practical limitations.

First, the number of local variational parameter sets grows with the dataset size.

Second, for a new observation $x_{\mathrm{new}}$, we generally need to solve a new optimization problem to infer its approximate posterior.

This can become computationally expensive when the dataset contains millions of observations.

### 6.2 Learning an Inference Network

**Amortized variational inference** addresses these limitations by replacing separately optimized local variational parameters with a shared inference network.

Instead of assigning each observation its own independent parameters $\phi_i$, we introduce a conditional distribution

$$
\boxed{
q_\phi(\mathbf z\mid x),
}
$$

where $\phi$ denotes parameters shared across all observations.

The inference network takes $x$ as input and predicts the parameters of its approximate posterior distribution.

For example, suppose we use a diagonal Gaussian variational family:

$$
\boxed{
q_\phi(\mathbf z\mid x)
=
\mathcal N_d
\left(
\mathbf z;
\boldsymbol\mu_\phi(x),
\operatorname{diag}
\left(\boldsymbol\sigma_\phi^2(x)\right)
\right).
}
$$

Here,

$$
\boldsymbol\mu_\phi(x)\in\mathbb R^d,
$$

and

$$
\boldsymbol\sigma_\phi^2(x)\in\mathbb R_{>0}^d.
$$

Both are outputs of the inference network.

A typical implementation uses a neural network with a shared representation and two output heads:

$$
h=f_\phi(x),
$$

$$
\boldsymbol\mu_\phi(x)=g_\phi^\mu(h),
$$

$$
\log\boldsymbol\sigma_\phi^2(x)
=
g_\phi^{\mathrm{var}}(h).
$$

The network predicts log-variances rather than unconstrained raw variances, ensuring positivity after exponentiation.

In the context of Variational Autoencoders, this inference network is called the **encoder**.

The optimization objective becomes

$$
\boxed{
\max_{\theta,\phi}
\sum_{i=1}^{n}
\mathcal L(\theta,\phi;x_i).
}
$$

where

$$
\mathcal L(\theta,\phi;x_i)
=
\mathbb E_{\mathbf z_i\sim q_\phi(\mathbf z_i\mid x_i)}
\left[
\log
\frac{p_\theta(x_i,\mathbf z_i)}
{q_\phi(\mathbf z_i\mid x_i)}
\right].
$$

The difference is that the same $\phi$ is used for every observation.

After training, approximate inference for a new observation requires only a forward pass through the network:

$$
\boxed{
x_{\mathrm{new}}
\longrightarrow
\left(
\boldsymbol\mu_\phi(x_{\mathrm{new}}),
\boldsymbol\sigma_\phi^2(x_{\mathrm{new}})
\right).
}
$$

This avoids solving a new variational optimization problem from scratch for each observation.

The method is called **amortized inference** because the computational cost of learning the inference procedure is shared across the entire dataset.

However, amortization introduces an additional restriction: the approximate posteriors must be representable by the shared inference network.

### 6.3 Variational Approximation Gap vs Amortization Gap

Amortized variational inference can introduce two conceptually different sources of inference error.

**Variational approximation gap**

The first limitation comes from the chosen variational family.

For example, if the true posterior is strongly correlated but the approximation is restricted to diagonal Gaussian distributions, the true posterior may not belong to the variational family.

Even perfect optimization cannot eliminate this representational limitation.

**Amortization gap**

The second limitation arises because the inference network may fail to predict the optimal variational parameters for each observation.

To distinguish these two effects, fix the generative model parameter $\theta$ and consider a particular observation $x$.

Let $\mathcal Q_x$ denote the chosen family of admissible approximate posterior distributions for that observation.

Define the optimal distribution within this family as

$$
\boxed{
q_x^*
=
\operatorname*{arg\,max}_{q\in\mathcal Q_x}
\mathcal L(q,\theta;x).
}
$$

Here,

$$
\mathcal L(q,\theta;x)
=
\mathbb E_{\mathbf z\sim q(\mathbf z)}
\left[
\log
\frac{p_\theta(x,\mathbf z)}
{q(\mathbf z)}
\right].
$$

Thus, $q_x^*$ represents the best possible approximation within the chosen variational family, assuming that the global optimum exists and is attained.

By comparison, the amortized inference network produces

$$
q_\phi(\mathbf z\mid x).
$$

The ELBO obtained from the inference network is

$$
\mathcal L(\theta,\phi;x)
=
\mathcal L(q_\phi(\cdot\mid x),\theta;x).
$$

The total inference gap is

$$
\log p_\theta(x)
-
\mathcal L(\theta,\phi;x).
$$

Adding and subtracting $\mathcal L(q_x^*,\theta;x)$ gives

$$
\boxed{
\begin{aligned}
&\log p_\theta(x)
-\mathcal L(\theta,\phi;x)\\[4pt]
&=
\underbrace{
\log p_\theta(x)
-
\mathcal L(q_x^*,\theta;x)
}_{\text{Variational approximation gap}}\\[6pt]
&\quad+
\underbrace{
\mathcal L(q_x^*,\theta;x)
-
\mathcal L(\theta,\phi;x)
}_{\text{Amortization gap}}.
\end{aligned}
}
$$

The first term measures the error caused by restricting the posterior approximation to the variational family $\mathcal Q_x$.

The second term measures how much worse the inference network performs compared with the best approximation available within that family.

If the inference network produces the locally optimal variational distribution,

$$
q_\phi(\mathbf z\mid x)=q_x^*(\mathbf z),
$$

then the amortization gap is zero.

However, the variational approximation gap may remain positive if the true posterior is not representable within $\mathcal Q_x$.

Conversely, even if the variational family is expressive enough to represent the true posterior, the inference network may fail to predict its optimal parameters.

Therefore, increasing the expressiveness of the variational family and improving the inference network address different sources of approximation error.

This distinction motivates more expressive posterior families, such as normalizing flows, as well as methods that improve amortized inference through additional local optimization.

## 7. The Reparameterization Trick

### 7.1 Why Is Sampling Difficult to Differentiate?

In amortized inference, the approximate posterior depends on neural network parameters $\phi$.

Our objective contains an expectation of the form

$$
\mathbb E_{z\sim q_\phi(z\mid x)}[f(z)].
$$

We would like to calculate

$$
\nabla_\phi
\mathbb E_{z\sim q_\phi(z\mid x)}[f(z)].
$$

However, samples drawn from $q_\phi(z\mid x)$ depend on $\phi$ through the sampling distribution.

If we sample $z$ and then treat it as a constant, ordinary backpropagation through $f(z)$ will not capture the effect of changing $\phi$ on the sampled values.

The reparameterization trick rewrites sampling as a deterministic transformation of parameter-independent random noise.

### 7.2 Gaussian Reparameterization

Suppose the encoder outputs

$$
q_\phi(z\mid x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

Instead of sampling $z$ directly, sample

$$
\epsilon\sim\mathcal N(0,I)
$$

and define

$$
\boxed{
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon.
}
$$

Here, $\odot$ denotes element-wise multiplication.

Why does this produce the correct distribution?

Because

$$
\mathbb E[z]
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\mathbb E[\epsilon]
=
\mu_\phi(x),
$$

and

$$
\operatorname{Cov}(z)
=
\operatorname{diag}(\sigma_\phi^2(x)).
$$

Since $z$ is an affine transformation of a Gaussian variable, it is Gaussian with the desired mean and covariance.

The crucial difference is that $\epsilon$ is drawn from a fixed distribution that does not depend on $\phi$.

Therefore,

$$
\begin{aligned}
&\nabla_\phi
\mathbb E_{z\sim q_\phi(z\mid x)}[f(z)]\\
&=
\nabla_\phi
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[
f\left(
\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
\right)
\right].
\end{aligned}
$$

Under suitable differentiability and integrability conditions, we can interchange differentiation and expectation:

$$
\boxed{
\begin{aligned}
&\nabla_\phi
\mathbb E_{z\sim q_\phi(z\mid x)}[f(z)]\\
&=
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[
\nabla_\phi
f\left(
\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
\right)
\right].
\end{aligned}
}
$$

Here $f$ is assumed not to depend explicitly on $\phi$ beyond its input $z$; otherwise its direct parameter derivative must also be included.

This allows the gradient to propagate through

$$
\phi
\longrightarrow
(\mu_\phi,\sigma_\phi)
\longrightarrow
z
\longrightarrow
f(z).
$$

### 7.3 Monte Carlo Gradient Estimation

The expectation over $\epsilon$ can be approximated using Monte Carlo samples.

Given independent samples

$$
\epsilon^{(1)},\ldots,\epsilon^{(L)}
\sim\mathcal N(0,I),
$$

we estimate

$$
\boxed{
\begin{aligned}
&\nabla_\phi
\mathbb E_{z\sim q_\phi(z\mid x)}[f(z)]\\
&\approx
\frac1L
\sum_{\ell=1}^{L}
\nabla_\phi
f\left(
\mu_\phi(x)+
\sigma_\phi(x)\odot\epsilon^{(\ell)}
\right).
\end{aligned}
}
$$

This is known as a **pathwise gradient estimator**.

In many VAE implementations, even $L=1$ sample per observation can work effectively when combined with minibatch training.

Reparameterization is especially convenient for continuous random variables such as Gaussian latent variables. Discrete latent variables generally require different estimators or continuous relaxations.

## 8. Summary and Connection to VAEs

Variational inference transforms intractable posterior inference into an optimization problem by introducing a tractable approximate posterior.

The central objective is to maximize the Evidence Lower Bound:

$$
\boxed{
\mathcal L(\theta,\phi;x)
=
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
}
$$

Mean-field variational inference simplifies posterior approximation through factorization, while coordinate ascent optimizes individual variational factors.

Amortized inference replaces separately optimized variational parameters with a shared inference network.

Finally, the reparameterization trick enables pathwise gradient estimation through continuous latent-variable sampling.

These ideas form the mathematical foundation of **Variational Autoencoders (VAEs)**, which combine an inference network and a generative model trained using the ELBO.

The next article develops the VAE architecture, training objective, and implementation in detail.

## References

1. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). *An Introduction to Variational Methods for Graphical Models.* Machine Learning.

2. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). *Variational Inference: A Review for Statisticians.* Journal of the American Statistical Association.

3. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.

4. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models.* ICML.

5. Kingma, D. P., & Welling, M. (2019). *An Introduction to Variational Autoencoders.* Foundations and Trends in Machine Learning.