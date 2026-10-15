---
layout: post
title: "M02 - Statistical Inference and Latent Variables"
date: 2026-10-13
mathjax: true
comments: true
---

**Prerequisites:** Probability distributions, conditional expectation, KL divergence.

**Aim:** Understand parameter estimation, Bayesian inference, latent-variable models, and the Expectation-Maximization (EM) algorithm.

## 1. Parametric Statistical Models

A parametric statistical model is a family of probability distributions indexed by a parameter $\theta$:

$$
\mathcal P=\{P_\theta:\theta\in\Theta\},
$$

where $\Theta$ is the parameter space.

If each distribution has a density, we write $p_\theta(x)$.

For example, the Gaussian location family is

$$
p_\theta(x)=\mathcal N(x;\theta,\sigma^2),
$$

where $\sigma^2$ is known and $\theta$ represents the unknown mean.

For a fixed $\theta$, $p_\theta$ specifies one probability distribution. Statistical inference aims to learn the unknown $\theta$ from observed data.

Suppose we observe

$$
\mathcal D=\{x_1,\ldots,x_n\},
$$

where $X_1,\ldots,X_n$ are independently and identically distributed according to some $P_{\theta_0}$.

Our goal is to estimate the unknown parameter $\theta_0$.

## 2. Maximum Likelihood Estimation

### 2.1 Likelihood and Log-Likelihood

Given observations $x_1,\ldots,x_n$, the likelihood function is

$$
L(\theta;\mathcal D)
=
\prod_{i=1}^{n}p_\theta(x_i).
$$

The likelihood treats the observed data as fixed and considers the expression as a function of $\theta$.

The maximum likelihood estimator (MLE) is

$$
\boxed{
\hat\theta_{\mathrm{MLE}}
=
\operatorname*{arg\,max}_{\theta\in\Theta}
L(\theta;\mathcal D).
}
$$

Since the logarithm is strictly increasing, maximizing the likelihood is equivalent to maximizing the log-likelihood:

$$
\boxed{
\hat\theta_{\mathrm{MLE}}
=
\operatorname*{arg\,max}_{\theta\in\Theta}
\sum_{i=1}^{n}\log p_\theta(x_i).
}
$$

The logarithm turns products into sums, making optimization easier.

### 2.2 Example: Gaussian Mean Estimation

Suppose

$$
X_1,\ldots,X_n
\overset{\mathrm{iid}}{\sim}
\mathcal N(\mu,\sigma^2),
$$

where $\sigma^2$ is known.

The log-likelihood is

$$
\ell(\mu)
=
-\frac n2\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}
\sum_{i=1}^{n}(x_i-\mu)^2.
$$

Differentiating with respect to $\mu$,

$$
\frac{\partial\ell}{\partial\mu}
=
\frac{1}{\sigma^2}
\sum_{i=1}^{n}(x_i-\mu).
$$

Setting the derivative to zero gives

$$
\sum_{i=1}^{n}(x_i-\hat\mu)=0.
$$

Therefore,

$$
\boxed{
\hat\mu_{\mathrm{MLE}}
=
\frac1n\sum_{i=1}^{n}x_i.
}
$$

The second derivative is

$$
\frac{\partial^2\ell}{\partial\mu^2}
=
-\frac n{\sigma^2}<0,
$$

so the critical point is the unique global maximizer.

Thus, the sample mean is the maximum likelihood estimator of the Gaussian mean.

## 3. Bayesian Inference and MAP Estimation

### 3.1 Bayesian Posterior

In frequentist estimation, $\theta$ is treated as an unknown but fixed parameter.

In Bayesian inference, we assign a prior distribution $p(\theta)$ to the parameter.

After observing data $\mathcal D$, Bayes' rule gives

$$
\boxed{
p(\theta\mid\mathcal D)
=
\frac{p(\mathcal D\mid\theta)p(\theta)}
{p(\mathcal D)}.
}
$$

Here:

- $p(\theta)$ is the prior.
- $p(\mathcal D\mid\theta)$ is the likelihood.
- $p(\theta\mid\mathcal D)$ is the posterior.
- $p(\mathcal D)$ is the marginal likelihood (evidence).

The evidence is obtained by integrating over all possible parameter values:

$$
p(\mathcal D)
=
\int p(\mathcal D\mid\theta)p(\theta)\,d\theta.
$$

### 3.2 Maximum a Posteriori Estimation

The maximum a posteriori estimator (MAP) is the posterior mode:

$$
\hat\theta_{\mathrm{MAP}}
=
\operatorname*{arg\,max}_{\theta}
p(\theta\mid\mathcal D).
$$

Since $p(\mathcal D)$ does not depend on $\theta$,

$$
\boxed{
\hat\theta_{\mathrm{MAP}}
=
\operatorname*{arg\,max}_{\theta}
\left[
\log p(\mathcal D\mid\theta)
+
\log p(\theta)
\right].
}
$$

Comparing MLE and MAP,

$$
\hat\theta_{\mathrm{MLE}}
=
\operatorname*{arg\,max}_{\theta}
\log p(\mathcal D\mid\theta),
$$

while MAP includes an additional log-prior term.

Thus, MAP estimation can be viewed as likelihood optimization with prior-based regularization.

### 3.3 Example: Gaussian Prior and Gaussian Likelihood

Suppose

$$
X_i\mid\mu
\overset{\mathrm{iid}}{\sim}
\mathcal N(\mu,\sigma^2),
$$

with prior

$$
\mu\sim\mathcal N(\mu_0,\tau^2).
$$

The posterior density is proportional to

$$
p(\mu\mid\mathcal D)
\propto
\exp\left[
-\frac{1}{2\sigma^2}
\sum_{i=1}^{n}(x_i-\mu)^2
-\frac{(\mu-\mu_0)^2}{2\tau^2}
\right].
$$

Collecting the terms involving $\mu$ and completing the square gives

$$
\mu\mid\mathcal D
\sim
\mathcal N(\mu_n,\tau_n^2),
$$

where

$$
\boxed{
\tau_n^2
=
\left(
\frac n{\sigma^2}
+
\frac1{\tau^2}
\right)^{-1}
}
$$

and

$$
\boxed{
\mu_n
=
\tau_n^2
\left(
\frac{n\bar x}{\sigma^2}
+
\frac{\mu_0}{\tau^2}
\right).
}
$$

The posterior mean is a precision-weighted average of the sample mean and the prior mean.

Because the posterior is Gaussian, its mode equals its mean:

$$
\hat\mu_{\mathrm{MAP}}=\mu_n.
$$

As $\tau^2\to\infty$, the prior becomes increasingly diffuse and

$$
\hat\mu_{\mathrm{MAP}}\to\bar x
=
\hat\mu_{\mathrm{MLE}}.
$$

### 3.4 Posterior Predictive Distribution

Bayesian inference also allows us to predict a new observation while accounting for parameter uncertainty.

The posterior predictive distribution is

$$
\boxed{
p(x_{\mathrm{new}}\mid\mathcal D)
=
\int
p(x_{\mathrm{new}}\mid\theta)
p(\theta\mid\mathcal D)\,d\theta.
}
$$

Instead of substituting a single parameter estimate, we average predictions over the posterior distribution.

This distinguishes Bayesian prediction from plug-in prediction using MLE or MAP.

For the Gaussian example above, with known observation variance $\sigma^2$,

$$
X_{\mathrm{new}}\mid\mathcal D
\sim
\mathcal N(\mu_n,\sigma^2+\tau_n^2).
$$

The predictive variance includes both observation noise $\sigma^2$ and remaining parameter uncertainty $\tau_n^2$.

## 4. Latent Variable Models

### 4.1 Introducing Latent Variables

A latent variable is a random variable that is not directly observed.

Suppose $X$ is observed and $Z$ is latent. A latent-variable model specifies the joint distribution

$$
\boxed{
p_\theta(x,z)
=
p_\theta(x\mid z)p_\theta(z).
}
$$

The distribution of the observed variable is obtained by marginalizing out $z$:

$$
\boxed{
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
}
$$

For discrete latent variables,

$$
p_\theta(x)=\sum_z p_\theta(x,z).
$$

Although the joint distribution may have a simple structure, its marginal distribution can be more complicated.

### 4.2 Observed-Data and Complete-Data Likelihood

Given independent observations $x_1,\ldots,x_n$, the observed-data log-likelihood is

$$
\ell(\theta)
=
\sum_{i=1}^{n}\log p_\theta(x_i).
$$

Using marginalization,

$$
\boxed{
\ell(\theta)
=
\sum_{i=1}^{n}
\log\int p_\theta(x_i,z_i)\,dz_i.
}
$$

If the latent variables $z_1,\ldots,z_n$ were also observed, we could instead use the complete-data log-likelihood:

$$
\ell_c(\theta)
=
\sum_{i=1}^{n}
\log p_\theta(x_i,z_i).
$$

The distinction is important:

- Observed-data likelihood uses only the available observations.
- Complete-data likelihood treats both observed and latent variables as known.

The difficulty in maximum likelihood estimation often comes from the logarithm of a sum or integral over the latent variables.

### 4.3 Example: Gaussian Mixture Model

Consider a mixture of $K$ Gaussian distributions:

$$
p_\theta(x)
=
\sum_{k=1}^{K}
\pi_k\mathcal N(x;\mu_k,\Sigma_k),
$$

where

$$
\pi_k\geq0,
\qquad
\sum_{k=1}^{K}\pi_k=1.
$$

Introduce a latent categorical variable $Z$ indicating the mixture component:

$$
P(Z=k)=\pi_k,
$$

$$
X\mid Z=k
\sim
\mathcal N(\mu_k,\Sigma_k).
$$

The joint distribution is

$$
p_\theta(x,Z=k)
=
\pi_k\mathcal N(x;\mu_k,\Sigma_k).
$$

Marginalizing $Z$ recovers the mixture density:

$$
p_\theta(x)
=
\sum_{k=1}^{K}
p_\theta(x,Z=k).
$$

The posterior probability that observation $x_i$ belongs to component $k$ is

$$
\boxed{
\gamma_{ik}
=
P_\theta(Z_i=k\mid X_i=x_i)
=
\frac{
\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)
}{
\sum_{j=1}^{K}
\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)
}.
}
$$

These posterior probabilities are called responsibilities.

Rather than assigning each observation to exactly one component, a mixture model allows soft assignments across components.

## 5. Expectation-Maximization (EM)

The **Expectation-Maximization (EM)** algorithm is an iterative method for maximum likelihood estimation in probabilistic models involving latent variables or missing data.

Its central idea is to replace a difficult marginal likelihood optimization problem with a sequence of simpler problems involving the complete-data log-likelihood.

EM alternates between two steps:

- **E-step (Expectation):** Infer the posterior distribution of latent variables under the current model parameters.
- **M-step (Maximization):** Update the model parameters by maximizing the expected complete-data log-likelihood.

### 5.1 Motivation: Why Do We Need EM?

Suppose we observe data $X=\{x_1,\ldots,x_n\}$, generated by a probabilistic model with parameters $\theta$ and unobserved latent variables $Z$.

Our objective is maximum likelihood estimation:

$$
\boxed{
\theta_{\mathrm{MLE}}
=
\operatorname*{arg\,max}_{\theta}
\log p_\theta(X).
}
$$

However, because $Z$ is not observed, the likelihood must be obtained by marginalizing over the latent variables:

$$
p_\theta(X)
=
\int p_\theta(X,Z)\,dZ.
$$

Therefore, the optimization problem becomes

$$
\boxed{
\operatorname*{arg\,max}_{\theta}
\log\int p_\theta(X,Z)\,dZ.
}
$$

For discrete latent variables, the integral is replaced by a summation.

The difficulty is that marginalization occurs **inside the logarithm**.

For example, in a Gaussian mixture model,

$$
p_\theta(x_i)
=
\sum_{k=1}^{K}
\pi_k\mathcal N(x_i;\mu_k,\Sigma_k),
$$

so the observed-data log-likelihood is

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log
\left[
\sum_{k=1}^{K}
\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)
\right].
$$

The logarithm of a sum introduces nonlinear dependencies between the mixture parameters. Although the observed-data log-likelihood can be optimized numerically, setting its derivatives to zero generally does not yield closed-form parameter estimates. EM addresses this difficulty by fixing the posterior responsibilities in the E-step, allowing simpler weighted parameter updates in the M-step.

In contrast, if the latent component assignments $Z$ were known, the complete-data log-likelihood would be considerably easier to optimize.

EM exploits this difference by using the current model to infer the latent variables and then updating the parameters as though the missing assignments were represented probabilistically.

Importantly, **EM is an optimization algorithm for maximum likelihood estimation**, not a different estimation principle.

### 5.2 E-Step: Posterior Inference

At iteration $t$, suppose the current parameter estimate is $\theta^{(t)}$.

The E-step first computes the posterior distribution of the latent variables:

$$
\boxed{
q^{(t)}(Z)
=
p_{\theta^{(t)}}(Z\mid X).
}
$$

This posterior describes how plausible different latent-variable configurations are, given the observed data and current model parameters.

Rather than selecting one particular latent-variable configuration, EM accounts for the uncertainty over $Z$.

It does so by constructing the **expected complete-data log-likelihood**:

$$
\boxed{
Q(\theta\mid\theta^{(t)})
=
\mathbb E_{p_{\theta^{(t)}}(Z\mid X)}
\left[
\log p_\theta(X,Z)
\right].
}
$$

For discrete latent variables, this can be written as

$$
Q(\theta\mid\theta^{(t)})
=
\sum_Z
p_{\theta^{(t)}}(Z\mid X)
\log p_\theta(X,Z).
$$

For continuous latent variables, the summation becomes an integral.

The posterior probabilities act as weights in the expectation.

**The two parameter arguments play different roles:**

- $\theta^{(t)}$ is the **fixed current parameter**, used to calculate the posterior distribution.
- $\theta$ is the **candidate parameter**, which will be optimized in the M-step.

Thus, although the posterior weights are determined by the current model, the Q-function remains a function of the new candidate parameters.

The reason for using this particular expectation is not arbitrary. As shown in Section 6, it arises naturally from maximizing a lower bound on the observed-data log-likelihood.

### 5.3 M-Step: Parameter Optimization

After constructing the Q-function, the M-step updates the model parameters:

$$
\boxed{
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
Q(\theta\mid\theta^{(t)}).
}
$$

During this optimization, the posterior distribution calculated in the E-step is held fixed.

Only the candidate model parameters $\theta$ are optimized.

The resulting parameter estimate $\theta^{(t+1)}$ is then used to calculate a new posterior distribution in the next E-step.

The algorithm repeats:

$$
\theta^{(0)}
\longrightarrow
Q(\theta\mid\theta^{(0)})
\longrightarrow
\theta^{(1)}
\longrightarrow
Q(\theta\mid\theta^{(1)})
\longrightarrow
\theta^{(2)}
\longrightarrow\cdots
$$

until a convergence criterion is satisfied, such as a sufficiently small change in the observed-data log-likelihood.

Under exact E- and M-steps, EM guarantees that the observed-data log-likelihood does not decrease between iterations. However, it does not generally guarantee convergence to a global maximum.

### 5.4 Example: EM for Gaussian Mixture Models

Consider a Gaussian mixture model with $K$ components:

$$
\boxed{
p_\theta(x)
=
\sum_{k=1}^{K}
\pi_k\mathcal N(x;\mu_k,\Sigma_k),
}
$$

where

$$
\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K},
$$

with mixture weights satisfying

$$
\pi_k\geq 0,
\qquad
\sum_{k=1}^{K}\pi_k=1.
$$

Introduce a discrete latent variable $Z_i\in\{1,\ldots,K\}$ indicating which component generates observation $x_i$.

The generative process is

$$
Z_i\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_K),
$$

followed by

$$
X_i\mid Z_i=k
\sim
\mathcal N(\mu_k,\Sigma_k).
$$

Since $Z_i$ is unobserved, the marginal distribution is obtained using the law of total probability:

$$
\begin{aligned}
p_\theta(x_i)
&=
\sum_{k=1}^{K}p_\theta(x_i,Z_i=k)\\
&=
\sum_{k=1}^{K}
p_\theta(Z_i=k)
p_\theta(x_i\mid Z_i=k)\\
&=
\sum_{k=1}^{K}
\pi_k\mathcal N(x_i;\mu_k,\Sigma_k).
\end{aligned}
$$

**E-step: Computing responsibilities**

Given the current parameters $\theta^{(t)}$, calculate the probability that observation $x_i$ belongs to component $k$:

$$
\gamma_{ik}^{(t)}
=
p_{\theta^{(t)}}(Z_i=k\mid X_i=x_i).
$$

Using Bayes' rule,

$$
\boxed{
\gamma_{ik}^{(t)}
=
\frac{
\pi_k^{(t)}
\mathcal N(x_i;\mu_k^{(t)},\Sigma_k^{(t)})
}{
\sum_{j=1}^{K}
\pi_j^{(t)}
\mathcal N(x_i;\mu_j^{(t)},\Sigma_j^{(t)})
}.
}
$$

These posterior probabilities are called **responsibilities**.

For every observation $i$,

$$
\sum_{k=1}^{K}\gamma_{ik}^{(t)}=1.
$$

A responsibility close to one indicates that the current model strongly associates the observation with that component.

Next, define the effective number of observations assigned to component $k$:

$$
\boxed{
N_k^{(t)}
=
\sum_{i=1}^{n}\gamma_{ik}^{(t)}.
}
$$

Unlike hard assignments, $N_k^{(t)}$ need not be an integer.

For the GMM, the complete-data log-likelihood can be written as

$$
\log p_\theta(X,Z)
=
\sum_{i=1}^{n}
\sum_{k=1}^{K}
\mathbb I(Z_i=k)
\left[
\log\pi_k+
\log\mathcal N(x_i;\mu_k,\Sigma_k)
\right].
$$

Taking the expectation under the current posterior replaces each indicator $\mathbb I(Z_i=k)$ with its posterior expectation $\gamma_{ik}^{(t)}$.

Therefore,

$$
\boxed{
Q(\theta\mid\theta^{(t)})
=
\sum_{i=1}^{n}
\sum_{k=1}^{K}
\gamma_{ik}^{(t)}
\left[
\log\pi_k+
\log\mathcal N(x_i;\mu_k,\Sigma_k)
\right].
}
$$

This is the objective optimized in the M-step.

**M-step: Updating the parameters**

Maximizing the Q-function with respect to the mixture weights, means, and covariance matrices yields the following updates.

Mixture weights:

$$
\boxed{
\pi_k^{(t+1)}
=
\frac{N_k^{(t)}}{n}.
}
$$

Means:

$$
\boxed{
\mu_k^{(t+1)}
=
\frac{1}{N_k^{(t)}}
\sum_{i=1}^{n}
\gamma_{ik}^{(t)}x_i.
}
$$

Covariance matrices:

$$
\boxed{
\Sigma_k^{(t+1)}
=
\frac{1}{N_k^{(t)}}
\sum_{i=1}^{n}
\gamma_{ik}^{(t)}
(x_i-\mu_k^{(t+1)})
(x_i-\mu_k^{(t+1)})^\top.
}
$$

These updates assume unconstrained full covariance matrices and a nondegenerate solution with $N_k^{(t)}>0$.

The means and covariance matrices are estimated using weighted observations, where the weights are the posterior responsibilities.

**Interpretation**

The E-step performs **soft assignment**: each observation is assigned a probability of belonging to every mixture component.

The M-step performs **weighted parameter estimation**: each component is updated according to how strongly it is responsible for the observations.

This differs from a hard-assignment approach, such as the assignment step in standard $K$-means, in which each observation belongs to exactly one cluster at each iteration.

One practical caveat is that unconstrained Gaussian mixture likelihoods can be unbounded when a component collapses around an observation and its covariance approaches singularity. Regularization or covariance constraints may therefore be necessary in practice.

---

## 6. EM as Lower-Bound Optimization

The previous section introduced the EM algorithm through its E- and M-steps.

However, an important theoretical question remains:

**Why does maximizing the expected complete-data log-likelihood help maximize the original observed-data likelihood?**

The answer comes from the **Evidence Lower Bound (ELBO)**.

EM can be interpreted as coordinate ascent on a lower bound of the observed-data log-likelihood. This perspective also reveals its close connection to variational inference.

### 6.1 Constructing a Lower Bound

Consider a latent-variable model with joint density $p_\theta(x,z)$.

The observed-data likelihood is obtained by marginalizing out the latent variable:

$$
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
$$

Our goal is to maximize

$$
\ell(\theta)=\log p_\theta(x).
$$

Introduce an auxiliary probability density $q(z)$.

Assume the relevant expectations are well-defined and, for the following equality, that $q$ is positive almost everywhere on the support of $p_\theta(x,z)$.

We can write

$$
\begin{aligned}
\log p_\theta(x)
&=
\log\int p_\theta(x,z)\,dz\\
&=
\log\int
q(z)\frac{p_\theta(x,z)}{q(z)}\,dz\\
&=
\log\mathbb E_q
\left[
\frac{p_\theta(x,Z)}{q(Z)}
\right].
\end{aligned}
$$

Since the logarithm is a concave function, Jensen's inequality gives

$$
\log\mathbb E_q[A]
\geq
\mathbb E_q[\log A]
$$

for a positive integrable random variable $A$.

Applying Jensen's inequality,

$$
\begin{aligned}
\log p_\theta(x)
&\geq
\mathbb E_q
\left[
\log\frac{p_\theta(x,Z)}{q(Z)}
\right]\\
&=
\mathbb E_q[\log p_\theta(x,Z)]
-
\mathbb E_q[\log q(Z)].
\end{aligned}
$$

Define the entropy of $q$ as

$$
H(q)
=
-\mathbb E_q[\log q(Z)].
$$

For continuous latent variables, this is differential entropy, provided it is well-defined.

We obtain

$$
\boxed{
\log p_\theta(x)
\geq
\underbrace{
\mathbb E_q[\log p_\theta(x,Z)]
+
H(q)
}_{\mathcal F(q,\theta)}.
}
$$

The function

$$
\boxed{
\mathcal F(q,\theta)
=
\mathbb E_q[\log p_\theta(x,Z)]
+
H(q)
}
$$

is called the **Evidence Lower Bound (ELBO)**.

Thus,

$$
\boxed{
\mathcal F(q,\theta)
\leq
\log p_\theta(x).
}
$$

Instead of directly maximizing the observed-data log-likelihood, we can attempt to maximize this lower bound.

The key question is how to choose $q$ so that the bound is sufficiently tight.

### 6.2 The KL Decomposition

The relationship between the ELBO and the true log-likelihood becomes clearer through Kullback–Leibler divergence.

By Bayes' rule,

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

Consider

$$
D_{\mathrm{KL}}
\left(
q(z)\|p_\theta(z\mid x)
\right)
=
\mathbb E_q
\left[
\log\frac{q(Z)}{p_\theta(Z\mid x)}
\right].
$$

Substituting Bayes' rule,

$$
\begin{aligned}
&D_{\mathrm{KL}}
\left(q(z)\|p_\theta(z\mid x)\right)\\
&=
\mathbb E_q
\left[
\log q(Z)
-
\log p_\theta(x,Z)
+
\log p_\theta(x)
\right]\\
&=
\mathbb E_q[\log q(Z)]
-
\mathbb E_q[\log p_\theta(x,Z)]
+
\log p_\theta(x).
\end{aligned}
$$

Rearranging gives the fundamental identity

$$
\boxed{
\log p_\theta(x)
=
\mathcal F(q,\theta)
+
D_{\mathrm{KL}}
\left(
q(z)\|p_\theta(z\mid x)
\right).
}
$$

Since KL divergence is non-negative,

$$
D_{\mathrm{KL}}(q\|p_\theta(z\mid x))\geq0,
$$

we recover

$$
\mathcal F(q,\theta)\leq\log p_\theta(x).
$$

More importantly, equality holds precisely when

$$
\boxed{
q(z)=p_\theta(z\mid x)
}
$$

almost everywhere.

This gives a mathematical explanation for the special role of the posterior distribution in EM.

**The exact posterior is the distribution that makes the ELBO equal to the observed-data log-likelihood at the current parameter.**

### 6.3 EM as Coordinate Ascent

We can now interpret EM as alternating maximization of the ELBO over two objects:

- The auxiliary distribution $q(z)$.
- The model parameters $\theta$.

The observed-data log-likelihood is fixed with respect to $q$ when $\theta$ is held constant.

Therefore, maximizing the ELBO over $q$ is equivalent to minimizing the KL divergence between $q$ and the posterior distribution.

**E-step: Optimizing the auxiliary distribution**

At iteration $t$, hold $\theta^{(t)}$ fixed and solve

$$
q^{(t+1)}
=
\operatorname*{arg\,max}_{q}
\mathcal F(q,\theta^{(t)}).
$$

From the KL decomposition,

$$
\log p_{\theta^{(t)}}(x)
=
\mathcal F(q,\theta^{(t)})
+
D_{\mathrm{KL}}
\left(
q\|p_{\theta^{(t)}}(z\mid x)
\right).
$$

The ELBO is maximized when the KL divergence is zero.

Thus,

$$
\boxed{
q^{(t+1)}(z)
=
p_{\theta^{(t)}}(z\mid x).
}
$$

At this choice,

$$
\boxed{
\mathcal F(q^{(t+1)},\theta^{(t)})
=
\log p_{\theta^{(t)}}(x).
}
$$

The lower bound becomes tight at the current parameter estimate.

This is the variational interpretation of the E-step.

**M-step: Optimizing the model parameters**

Now hold $q^{(t+1)}$ fixed and optimize

$$
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
\mathcal F(q^{(t+1)},\theta).
$$

Expanding the ELBO,

$$
\begin{aligned}
\mathcal F(q^{(t+1)},\theta)
={}&
\mathbb E_{q^{(t+1)}}
[\log p_\theta(x,Z)]\\
&+
H(q^{(t+1)}).
\end{aligned}
$$

Since $q^{(t+1)}$ is fixed, the entropy term does not depend on $\theta$.

Therefore,

$$
\boxed{
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
\mathbb E_{q^{(t+1)}}
[\log p_\theta(x,Z)].
}
$$

This is exactly the M-step introduced in Section 5.

**Connection to the Q-function**

Recall that the E-step uses

$$
q^{(t+1)}(z)
=
p_{\theta^{(t)}}(z\mid x).
$$

Consequently,

$$
\begin{aligned}
\mathcal F(q^{(t+1)},\theta)
&=
\mathbb E_{p_{\theta^{(t)}}(z\mid x)}
[\log p_\theta(x,Z)]\\
&\quad+
H(q^{(t+1)}).
\end{aligned}
$$

By the definition of the Q-function,

$$
\boxed{
\mathcal F(q^{(t+1)},\theta)
=
Q(\theta\mid\theta^{(t)})
+
H(q^{(t+1)}).
}
$$

Since the entropy term is independent of the candidate parameter $\theta$,

$$
\boxed{
\operatorname*{arg\,max}_\theta
\mathcal F(q^{(t+1)},\theta)
=
\operatorname*{arg\,max}_\theta
Q(\theta\mid\theta^{(t)}).
}
$$

This explains why EM uses the expected complete-data log-likelihood as its optimization objective.

**The Q-function is not an arbitrary construction. It is precisely the parameter-dependent part of the ELBO when the auxiliary distribution is fixed to the current posterior.**

Thus, EM can be interpreted as coordinate ascent on $\mathcal F(q,\theta)$:

1. Optimize $q$ while holding $\theta$ fixed.
2. Optimize $\theta$ while holding $q$ fixed.
3. Repeat.

### 6.4 Why EM Increases the Likelihood

The lower-bound interpretation provides a direct proof of EM's monotonicity property.

At iteration $t$, the E-step selects the exact posterior:

$$
q^{(t+1)}(z)
=
p_{\theta^{(t)}}(z\mid x).
$$

This makes the lower bound tight:

$$
\boxed{
\mathcal F(q^{(t+1)},\theta^{(t)})
=
\log p_{\theta^{(t)}}(x).
}
$$

The M-step then maximizes the ELBO with respect to $\theta$.

Therefore,

$$
\mathcal F(q^{(t+1)},\theta^{(t+1)})
\geq
\mathcal F(q^{(t+1)},\theta^{(t)}).
$$

Since the ELBO is always a lower bound on the log-likelihood,

$$
\log p_{\theta^{(t+1)}}(x)
\geq
\mathcal F(q^{(t+1)},\theta^{(t+1)}).
$$

Combining these relationships,

$$
\begin{aligned}
\log p_{\theta^{(t+1)}}(x)
&\geq
\mathcal F(q^{(t+1)},\theta^{(t+1)})\\
&\geq
\mathcal F(q^{(t+1)},\theta^{(t)})\\
&=
\log p_{\theta^{(t)}}(x).
\end{aligned}
$$

Hence,

$$
\boxed{
\log p_{\theta^{(t+1)}}(x)
\geq
\log p_{\theta^{(t)}}(x).
}
$$

This proves that **exact EM produces a non-decreasing sequence of observed-data log-likelihood values**.

In fact, the M-step does not need to find the exact maximizer to preserve this property. It is sufficient to increase, or at least not decrease, the Q-function. This leads to **Generalized EM (GEM)** algorithms.

However, monotonic likelihood improvement does not guarantee that EM reaches the global maximum.

Depending on the model and initialization, EM may converge to a local optimum or another stationary point under suitable regularity conditions. In some models, such as unconstrained Gaussian mixtures, likelihood singularities can also lead to degenerate parameter sequences.

### 6.5 Connection to Variational Inference

The lower-bound interpretation reveals a close relationship between EM and **Variational Inference (VI)**.

Both methods use an auxiliary distribution $q(z)$ to construct and optimize the ELBO:

$$
\mathcal F(q,\theta)
=
\mathbb E_q[\log p_\theta(x,Z)]
+
H(q).
$$

The key difference lies in how the E-step is performed.

**Exact EM**

In standard EM, the auxiliary distribution is chosen to be the exact posterior:

$$
\boxed{
q^{(t+1)}(z)
=
p_{\theta^{(t)}}(z\mid x).
}
$$

This makes the ELBO tight at the current parameter.

**Variational EM**

In models where the exact posterior is computationally intractable, we can restrict $q$ to a tractable variational family $\mathcal Q$.

The E-step becomes

$$
\boxed{
q^{(t+1)}
=
\operatorname*{arg\,max}_{q\in\mathcal Q}
\mathcal F(q,\theta^{(t)}).
}
$$

If the exact posterior does not belong to $\mathcal Q$, the KL divergence generally cannot be reduced to zero.

Therefore, the ELBO may remain strictly below the log-likelihood:

$$
\mathcal F(q^{(t+1)},\theta^{(t)})
<
\log p_{\theta^{(t)}}(x).
$$

The M-step still optimizes the model parameters using the current variational distribution:

$$
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
\mathcal F(q^{(t+1)},\theta).
$$

Thus, variational EM optimizes a potentially non-tight lower bound, rather than using the exact posterior at each iteration.

While exact EM guarantees non-decreasing observed-data likelihood under exact updates, variational EM generally guarantees improvement of the optimized ELBO, not necessarily of the true observed-data likelihood.

This connection extends naturally to modern latent-variable generative models, including **Variational Autoencoders (VAEs)**, where an inference network is used to approximate the posterior distribution and the ELBO is optimized using gradient-based methods.

**Summary**

The essential relationships are

$$
\boxed{
\underbrace{\log p_\theta(x)}_{\text{Observed-data log-likelihood}}
=
\underbrace{\mathcal F(q,\theta)}_{\text{ELBO}}
+
\underbrace{
D_{\mathrm{KL}}(q\|p_\theta(z\mid x))
}_{\text{Approximation gap}}.
}
$$

Standard EM chooses the exact posterior in the E-step to eliminate the approximation gap at the current parameters, then updates the parameters by maximizing the Q-function.

Variational inference generalizes this idea by optimizing over a restricted family of approximate posterior distributions when exact inference is difficult.

## 7. Summary and Connections

| Mathematical idea     | Key result                                                   |
| :-------------------- | :----------------------------------------------------------- |
| Parametric model      | $\{p_\theta:\theta\in\Theta\}$                               |
| MLE                   | Maximize the observed-data likelihood                        |
| MAP                   | Maximize likelihood times prior                              |
| Bayesian inference    | Infer a posterior distribution over parameters               |
| Posterior prediction  | Integrate predictions over parameter uncertainty             |
| Latent-variable model | $p_\theta(x)=\int p_\theta(x,z)\,dz$                         |
| E-step                | Compute the latent-variable posterior                        |
| M-step                | Maximize the expected complete-data log-likelihood           |
| EM lower bound        | $\log p_\theta(x)=\mathcal F(q,\theta)+D_{\mathrm{KL}}(q\|p_\theta(z\mid x))$ |

The key insight is that EM transforms a difficult marginal-likelihood optimization problem into alternating posterior inference and parameter optimization.

This lower-bound interpretation provides the mathematical foundation for the variational inference methods studied in M03.

## References

- Bishop, *Pattern Recognition and Machine Learning*, Chapter 9.
- Murphy, *Probabilistic Machine Learning: An Introduction*.
- Dempster, Laird, and Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*.