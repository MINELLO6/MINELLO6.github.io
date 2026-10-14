---
layout: post
title: "M02 - Statistical Inference and Latent Variable Models"
date: 2026-10-13
mathjax: true
comments: true
---

# Statistical Inference and Latent Variable Models

**Prerequisite:** [M01 - Probability and Information Theory]({{ '/2026/10/13/probability-information-theory/' | relative_url }}).  
**Aim:** connect classical estimation, Bayesian inference, latent variables, and EM to the learning objectives of deep generative models.

## 1. Maximum likelihood estimation

Given i.i.d. observations $D=\{x_i\}_{i=1}^n$ modeled by $p_\theta$, maximum likelihood estimates

$$
\widehat\theta_{\rm MLE}=\arg\max_\theta\sum_{i=1}^n\log p_\theta(x_i).
$$

Why use a log? It converts a product of likelihoods into a sum and has the same maximizers since logarithm is increasing. Under an idealized population distribution $p_{\rm data}$,

$$
\begin{aligned}
\mathbb E_{p_{\rm data}}[-\log p_\theta(X)]
&=H(p_{\rm data},p_\theta)\\
&=H(p_{\rm data})+D_{\rm KL}(p_{\rm data}\|p_\theta).
\end{aligned}
$$

Therefore population MLE minimizes forward KL when the entropy and KL are defined. For flexible generative models, the challenge is often *evaluating* $p_\theta(x)$, not merely sampling from it.

### 1.1 Gaussian example

For $x_i\overset{\rm iid}{\sim}\mathcal N(\mu,\sigma^2)$ with fixed $\sigma^2$,

$$
\ell(\mu)=-\frac{1}{2\sigma^2}\sum_i(x_i-\mu)^2+\text{constant},
\qquad
\frac{\partial\ell}{\partial\mu}=\frac{1}{\sigma^2}\sum_i(x_i-\mu).
$$

Setting the derivative to zero gives $\widehat\mu=\frac1n\sum_i x_i$. This demonstrates how maximizing Gaussian likelihood becomes minimizing squared error.

## 2. Bayesian inference and MAP

A Bayesian model places a prior $p(\theta)$ on parameters and updates using data:

$$
p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)},
\qquad
p(D)=\int p(D\mid\theta)p(\theta)\,d\theta.
$$

MAP estimation chooses the mode of the posterior:

$$
\widehat\theta_{\rm MAP}=\arg\max_\theta\left[\log p(D\mid\theta)+\log p(\theta)\right].
$$

For a zero-mean isotropic Gaussian prior, its negative log-density contributes an $L^2$ penalty. **Caution:** the MAP point estimate is not the complete Bayesian posterior; its value can also depend on parameterization.

Predicting new data fully Bayesianly requires averaging parameter uncertainty:

$$
p(x_\star\mid D)=\int p(x_\star\mid\theta)p(\theta\mid D)\,d\theta.
$$

**Distinction worth preserving:** a VAE has a prior over sample-specific latent variables $z$, but its network weights $\theta,\phi$ are typically point-estimated. This does not make the network weights automatically Bayesian.

## 3. Latent-variable generative models

A latent-variable model specifies a joint distribution

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z).
$$

Generation proceeds in the order $z\sim p(z)$ and then $x\sim p_\theta(x\mid z)$. To evaluate the likelihood of observed data, however, we need

$$
p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz.
$$

The posterior over the latent code is

$$
p_\theta(z\mid x)=\frac{p(z)p_\theta(x\mid z)}{\int p(z')p_\theta(x\mid z')\,dz'}.
$$

**Three different tasks:** (1) sampling $x$, (2) evaluating $p_\theta(x)$, and (3) inferring $z\mid x$. One can be easy while the others are computationally hard.

## 4. Gaussian mixture model: a worked latent-variable example

Let $Z\in\{1,\ldots,K\}$ be a component assignment with $P(Z=k)=\pi_k$. Let

$$
X\mid(Z=k)\sim\mathcal N(\mu_k,\Sigma_k).
$$

The marginal data distribution is

$$
p_\theta(x)=\sum_{k=1}^K\pi_k\mathcal N(x;\mu_k,\Sigma_k).
$$

Unlike the complete-data likelihood, $\log\sum_k\pi_k\mathcal N(x;\mu_k,\Sigma_k)$ mixes components inside a logarithm, making optimization harder.

## 5. Expectation-Maximization (EM)

The Expectation-Maximization algorithm is an iterative optimization method for maximum likelihood estimation with missing or latent variables. Its purpose is **to estimate the model parameters**, not merely to infer latent variables.

### 5.1 Why do we need EM?

For observed data $X$ and latent variables $Z$, the maximum likelihood objective is

$$
\theta_{\mathrm{MLE}}=\operatorname*{arg\,max}_{\theta}\log p_\theta(X)
=\operatorname*{arg\,max}_{\theta}\log\int p_\theta(X,Z)\,dZ.
$$

Marginalizing over $Z$ *inside* a logarithm can make optimization difficult. For instance, in a Gaussian mixture model,

$$
\ell(\theta)=\sum_{i=1}^{n}\log\left[\sum_{k=1}^K\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)\right].
$$

The sum inside the logarithm couples different mixture components. This **does not mean direct numerical optimization is impossible**; rather, setting the derivatives to zero generally produces coupled, nonlinear equations without convenient closed-form solutions. EM introduces an auxiliary objective whose parameter updates can be simpler.

### 5.2 E-step: posterior inference

Given the current parameters $\theta^{(t)}$, compute the exact latent-variable posterior

$$
q^{(t+1)}(Z)=p_{\theta^{(t)}}(Z\mid X).
$$

Use it to construct the expected complete-data log-likelihood:

$$
\boxed{
Q(\theta\mid\theta^{(t)})
=\mathbb E_{p_{\theta^{(t)}}(Z\mid X)}
[\log p_\theta(X,Z)].
}
$$

In the expectation, **$\theta^{(t)}$ is fixed** and determines the posterior weights; **$\theta$ is the candidate parameter** to be optimized next. For a discrete $Z$, the expectation is a posterior-weighted sum over possible latent assignments.

### 5.3 M-step: parameter optimization

Fix the posterior weights and maximize the Q-function:

$$
\boxed{
\theta^{(t+1)}=\operatorname*{arg\,max}_{\theta}Q(\theta\mid\theta^{(t)}).
}
$$

Repeat the E-step and M-step until convergence. Under exact steps, the observed-data log-likelihood does not decrease, though global optimality is not guaranteed.

### 5.4 Worked example: Gaussian mixture models

Let $Z_i\in\{1,\ldots,K\}$ indicate the mixture component that generated observation $x_i$:

$$
P(Z_i=k)=\pi_k,\qquad X_i\mid Z_i=k\sim\mathcal N(\mu_k,\Sigma_k).
$$

Marginalization gives

$$
p_\theta(x_i)=\sum_{k=1}^K\pi_k\mathcal N(x_i;\mu_k,\Sigma_k).
$$

By Bayes' rule, the **responsibility** of component $k$ for observation $i$ is

$$
\boxed{
\gamma_{ik}^{(t)}
=P(Z_i=k\mid x_i,\theta^{(t)})
=\frac{\pi_k^{(t)}\mathcal N(x_i;\mu_k^{(t)},\Sigma_k^{(t)})}
{\sum_{j=1}^K\pi_j^{(t)}\mathcal N(x_i;\mu_j^{(t)},\Sigma_j^{(t)})}.
}
$$

The complete-data log-likelihood can be expressed as

$$
\log p_\theta(X,Z)
=\sum_{i=1}^{n}\sum_{k=1}^K\mathbb I(Z_i=k)
\left[\log\pi_k+\log\mathcal N(x_i;\mu_k,\Sigma_k)\right].
$$

Since $\mathbb E[\mathbb I(Z_i=k)\mid X,\theta^{(t)}]=\gamma_{ik}^{(t)}$, the Q-function is

$$
Q(\theta\mid\theta^{(t)})
=\sum_{i=1}^n\sum_{k=1}^K\gamma_{ik}^{(t)}
\left[\log\pi_k+\log\mathcal N(x_i;\mu_k,\Sigma_k)\right].
$$

Define $N_k^{(t)}=\sum_i\gamma_{ik}^{(t)}$. Maximizing this Q-function gives the M-step:

$$
\boxed{
\begin{aligned}
\pi_k^{(t+1)}&=\frac{N_k^{(t)}}{n},\\
\mu_k^{(t+1)}&=\frac{1}{N_k^{(t)}}\sum_i\gamma_{ik}^{(t)}x_i,\\
\Sigma_k^{(t+1)}&=\frac{1}{N_k^{(t)}}\sum_i\gamma_{ik}^{(t)}
(x_i-\mu_k^{(t+1)})(x_i-\mu_k^{(t+1)})^\top.
\end{aligned}
}
$$

These formulas assume nondegenerate, unconstrained full covariance matrices and $N_k^{(t)}>0$. EM uses **soft** assignments rather than assigning every observation to exactly one cluster.

## 6. EM as Lower-Bound Optimization

Why does maximizing $Q$ help maximize the *marginal* likelihood? The answer is an evidence lower bound (ELBO).

### 6.1 Deriving the lower bound

For an auxiliary density $q(z)$ with appropriate support and finite relevant expectations,

$$
\begin{aligned}
\log p_\theta(x)
&=\log\int q(z)\frac{p_\theta(x,z)}{q(z)}\,dz\\
&=\log\mathbb E_q\left[\frac{p_\theta(x,Z)}{q(Z)}\right]\\
&\geq\mathbb E_q[\log p_\theta(x,Z)]-\mathbb E_q[\log q(Z)].
\end{aligned}
$$

The inequality follows from Jensen's inequality and the concavity of $\log$. Define

$$
\boxed{
\mathcal F(q,\theta)=\mathbb E_q[\log p_\theta(x,Z)]+H(q),
\qquad H(q)=-\mathbb E_q[\log q(Z)].
}
$$

Thus $\mathcal F(q,\theta)\leq\log p_\theta(x)$.

### 6.2 The KL decomposition

Bayes' rule implies the exact identity

$$
\boxed{
\log p_\theta(x)=\mathcal F(q,\theta)
+D_{\mathrm{KL}}\bigl(q(z)\|p_\theta(z\mid x)\bigr).
}
$$

Because KL divergence is nonnegative, the ELBO is a lower bound. It is **tight** when $q(z)=p_\theta(z\mid x)$ almost everywhere.

### 6.3 EM as coordinate ascent

There are two optimization objects: the auxiliary distribution $q$ and the model parameters $\theta$.

**E-step.** With $\theta^{(t)}$ fixed, maximize $\mathcal F(q,\theta^{(t)})$ over all distributions $q$. The log-likelihood is constant with respect to $q$, so maximizing the ELBO is equivalent to minimizing KL divergence to the current posterior. Thus

$$
\boxed{
q^{(t+1)}(z)=p_{\theta^{(t)}}(z\mid x),\quad
\mathcal F(q^{(t+1)},\theta^{(t)})=\log p_{\theta^{(t)}}(x).
}
$$

**M-step.** Fix $q^{(t+1)}$ and maximize $\mathcal F(q^{(t+1)},\theta)$ over $\theta$. Since its entropy is constant in this step,

$$
\begin{aligned}
\theta^{(t+1)}
&=\operatorname*{arg\,max}_{\theta}
\mathbb E_{q^{(t+1)}}[\log p_\theta(x,Z)]\\
&=\boxed{\operatorname*{arg\,max}_{\theta}Q(\theta\mid\theta^{(t)})}.
\end{aligned}
$$

In fact, $\mathcal F(q^{(t+1)},\theta)=Q(\theta\mid\theta^{(t)})+H(q^{(t+1)})$: **the Q-function is precisely the parameter-dependent part of the ELBO.**

### 6.4 Why exact EM increases likelihood

Combining lower-bound validity, the M-step improvement, and the E-step equality:

$$
\boxed{
\begin{aligned}
\log p_{\theta^{(t+1)}}(x)
&\geq \mathcal F(q^{(t+1)},\theta^{(t+1)})\\
&\geq \mathcal F(q^{(t+1)},\theta^{(t)})\\
&=\log p_{\theta^{(t)}}(x).
\end{aligned}
}
$$

Therefore exact EM generates a **non-decreasing** sequence of observed-data log-likelihoods. A **generalized EM** (GEM) step need only improve or preserve $Q$, not maximize it exactly. The monotonicity result does not imply global optimality: EM may converge to a local optimum or another stationary point under suitable conditions. For unrestricted Gaussian mixtures, a component covariance can collapse around an observation, leading to likelihood singularities and degenerate parameter sequences.

### 6.5 Connection to variational inference

Exact EM chooses the exact posterior in every E-step. When this posterior is intractable, **variational EM** instead restricts the auxiliary distribution to a tractable family $\mathcal Q$:

$$
\boxed{
q^{(t+1)}
=\operatorname*{arg\,max}_{q\in\mathcal Q}\mathcal F(q,\theta^{(t)}).
}
$$

If the true posterior is not in $\mathcal Q$, the ELBO need not be tight, and improving it does **not** guarantee improvement of the true likelihood. Variational EM instead monotonically improves the ELBO when its coordinate updates are performed exactly.

**Amortized variational inference**, as used in a variational autoencoder (VAE), learns a shared encoder $q_\phi(z\mid x)$ that predicts an approximate posterior for each sample. A VAE usually trains encoder and decoder with stochastic gradients rather than exact alternating E/M maximization.

M03 will develop variational inference, the two directions of KL divergence, mean-field updates, reparameterization, and the VAE objective in detail.


## 7. A compact NumPy implementation of GMM EM

```python
import numpy as np

rng = np.random.default_rng(42)
x = np.r_[rng.normal(-2, 0.7, 200), rng.normal(2, 0.8, 300)]
mu = np.array([-1.0, 1.0])
var = np.ones(2)
pi = np.ones(2) / 2

for _ in range(50):
    density = np.exp(-0.5 * (x[:, None] - mu)**2 / var) / np.sqrt(2*np.pi*var)
    weighted = density * pi
    gamma = weighted / weighted.sum(axis=1, keepdims=True)  # E-step
    Nk = gamma.sum(axis=0)
    pi = Nk / len(x)
    mu = (gamma * x[:, None]).sum(axis=0) / Nk
    var = (gamma * (x[:, None] - mu)**2).sum(axis=0) / Nk  # M-step

print("means:", mu)
print("weights:", pi)
```

For a numerically robust implementation, evaluate mixture probabilities in log space, regularize nearly singular covariances, and compute the M-step variance using the same responsibilities and the newly updated mean (as above).

## 8. What to remember before M03

- MLE fits the *marginal* model to observed data.
- In latent-variable models, marginalization and posterior inference are often harder than sampling.
- EM alternates inference about $z$ and parameter learning.
- Its lower bound is an ELBO; variational inference replaces the exact E-step with a tractable approximation.

**Self-check:** Derive the GMM responsibility formula from Bayes' rule. Explain why Jensen's inequality appears in EM and why an approximate E-step may forfeit monotonic improvement of the true log-likelihood.

## References

- Dempster, Laird, and Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977).
- Bishop, *Pattern Recognition and Machine Learning*, Chapter 9.
- Murphy, *Probabilistic Machine Learning: An Introduction*.
- Neal and Hinton, *A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants* (1998).