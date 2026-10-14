---
layout: post
title: "M01 - Probability and Information Theory for Generative Modeling"
date: 2026-10-13
mathjax: true
comments: true
typora-root-url: F:/Blog-Backup/
---


**Prerequisites:** calculus, elementary probability.  
**Aim:** understand probability distributions and the information-theoretic objectives used in VAEs, autoregressive models, diffusion models, and normalizing flows.


## 1. Joint, Marginal, and Conditional Distributions

### 1.1 Joint Distribution

The joint distribution $p(x,z)$ describes the probability
behavior of two random variables $X$ and $Z$ together.

By the product rule,

$$
p(x,z)
=
p(x\mid z)p(z)
=
p(z\mid x)p(x).
$$

This identity allows us to express a joint distribution
using a marginal distribution and a conditional distribution.

### 1.2 Marginal Distribution

A marginal distribution describes one variable without
conditioning on the other.

Starting from the joint distribution $p(x,z)$,
we obtain $p(x)$ by summing or integrating out $z$.

For discrete variables,

$$
p(x)=\sum_z p(x,z).
$$

For continuous variables,

$$
p(x)=\int p(x,z)\,dz.
$$

Using the product rule,

$$
\boxed{
p(x)=\int p(x\mid z)p(z)\,dz.
}
$$

This is also known as the law of total probability
in its continuous form.

Intuitively, marginalization combines the contributions
from all possible values of $z$.

### 1.3 Conditional Distribution

A conditional distribution describes the behavior of
one variable when another variable is known.

For example,

$$
p(x\mid z)
=
\frac{p(x,z)}{p(z)},
$$

provided $p(z)>0$.

Unlike the marginal distribution $p(x)$,
the conditional distribution depends on the given value of $z$.

If $X$ and $Z$ are independent, then

$$
p(x\mid z)=p(x).
$$

Thus, conditioning only changes a distribution when
the conditioning variable provides information
about the variable of interest.

### 1.4 Bayes' Rule

The product rule gives two expressions for the
same joint distribution:

$$
p(x,z)=p(x\mid z)p(z)
$$

and

$$
p(x,z)=p(z\mid x)p(x).
$$

Equating them yields Bayes' rule:

$$
\boxed{
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
}
$$

The terms have standard interpretations:

- $p(z)$: prior distribution
- $p(x\mid z)$: likelihood
- $p(z\mid x)$: posterior distribution
- $p(x)$: marginal likelihood (evidence)

Bayes' rule tells us how the distribution of $Z$
changes after observing $X=x$.

### 1.5 Latent Variables

A latent variable is a random variable that is not
directly observed.

Suppose $X$ is observed and $Z$ is latent.
Their joint distribution can be factorized as

$$
p(x,z)=p(x\mid z)p(z).
$$

Even though $Z$ is not observed, the marginal
distribution of $X$ is still well-defined:

$$
p(x)
=
\int p(x\mid z)p(z)\,dz.
$$

Conversely, observing $X=x$ allows us to describe
our uncertainty about $Z$ through the posterior:

$$
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
$$

These two operations play different roles:

- **Marginalization:** eliminate an unobserved variable.
- **Conditioning:** update a distribution using observed information.

Both are fundamental operations in probabilistic modeling.


## 2. Expectations and Conditional Expectations
### 2.1 Expectation

The expectation of a random variable $X$ is its
probability-weighted average:

$$
\mathbb E[X]
=
\sum_x x\,p(x)
\quad\text{or}\quad
\mathbb E[X]
=
\int x\,p(x)\,dx.
$$

More generally, for a function $h$,

$$
\mathbb E[h(X)]
=
\int h(x)p(x)\,dx,
$$

provided the expectation exists.

### 2.2 Conditional Expectation

Conditional expectation describes the expected value
of a random variable given some observed information.

For continuous random variables,

$$
\boxed{
\mathbb E[Y\mid X=x]
=
\int y\,p(y\mid x)\,dy.
}
$$

Unlike $\mathbb E[Y]$, the conditional expectation
depends on the observed value $x$.

Consequently, $\mathbb E[Y\mid X]$ is itself
a random variable and a function of $X$.


### 2.3 Law of Total Expectation (Tower Property)

The law of total expectation states that

$$
\boxed{
\mathbb E[\mathbb E[Y\mid X]]
=
\mathbb E[Y].
}
$$

In words, averaging the conditional expectations over
the distribution of $X$ recovers the unconditional expectation.

**Proof.**

For continuous random variables, by the definition
of conditional expectation,

$$
\mathbb E[Y\mid X=x]
=
\int y\,p(y\mid x)\,dy.
$$

Taking expectation over $X$,

$$
\begin{aligned}
\mathbb E[\mathbb E[Y\mid X]]
&=
\int \mathbb E[Y\mid X=x]p(x)\,dx\\
&=
\int\int y\,p(y\mid x)p(x)\,dy\,dx\\
&=
\int y\left[\int p(x,y)\,dx\right]dy\\
&=
\int y\,p(y)\,dy\\
&=
\mathbb E[Y].
\end{aligned}
$$

The key step uses the product rule
$p(x,y)=p(y\mid x)p(x)$ and marginalization.

**Generalization.**

Conditioning on $(X,Z)$ and then averaging out $Z$
gives the expectation conditional on $X$:

$$
\boxed{
\mathbb E[\mathbb E[Y\mid X,Z]\mid X]
=
\mathbb E[Y\mid X].
}
$$

**Proof.**

For continuous random variables, conditioning on $X=x$ gives

$$
\begin{aligned}
&\mathbb E[\mathbb E[Y\mid X,Z]\mid X=x]\\
&=\int \mathbb E[Y\mid X=x,Z=z]p(z\mid x)\,dz\\
&=\int\int y\,p(y\mid x,z)p(z\mid x)\,dy\,dz\\
&=\int y\left[\int p(y,z\mid x)\,dz\right]dy\\
&=\int y\,p(y\mid x)\,dy\\
&=\mathbb E[Y\mid X=x].
\end{aligned}
$$


### 2.4 Conditional Expectation Minimizes Squared Error

An important property of conditional expectation is
that it minimizes mean squared prediction error.

Let

$$
m(X)=\mathbb E[Y\mid X].
$$

For any predictor $f(X)$ with finite second moment,

$$
\begin{aligned}
\mathbb E[(Y-f(X))^2]
&=
\mathbb E[(Y-m(X)+m(X)-f(X))^2]\\
&=
\mathbb E[(Y-m(X))^2]\\
&\quad+
\mathbb E[(m(X)-f(X))^2].
\end{aligned}
$$

The cross term vanishes because

$$
\mathbb E[Y-m(X)\mid X]=0.
$$


Since the second term is non-negative, the mean squared
error is minimized when $f(X)=m(X)$. Therefore,

$$
\boxed{
f^*
=
\operatorname*{arg\,min}_{f}
\mathbb E\left[(Y-f(X))^2\right],
\qquad
f^*(X)=\mathbb E[Y\mid X].
}
$$

The minimizer is unique up to almost-sure equality,
over all square-integrable measurable functions of $X$.


## 3. Entropy and Cross-Entropy

### 3.1 Shannon Entropy

Entropy measures the uncertainty of a random variable.

For a discrete random variable $X$ with probability mass
function $p(x)$, the Shannon entropy is

$$
\boxed{
H(X)=-\sum_x p(x)\log p(x).
}
$$

A distribution concentrated on one outcome has zero entropy,
while a uniform distribution over $n$ outcomes has
entropy $\log n$.

We use natural logarithms, so entropy is measured in
**nats** rather than bits.

### 3.2 Differential Entropy

For a continuous random variable with density $p(x)$,
the analogous quantity is differential entropy:

$$
\boxed{
h(X)=-\int p(x)\log p(x)\,dx.
}
$$

Unlike discrete entropy, differential entropy can be
negative and depends on the scale of the variable.

For example, if $Y=aX$ with $a\neq 0$,

$$
h(Y)=h(X)+\log|a|.
$$

**Gaussian example.**

For $X\sim\mathcal N(\mu,\sigma^2)$,

$$
p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

Taking logarithms,

$$
\log p(x)
=
-\frac12\log(2\pi\sigma^2)
-\frac{(x-\mu)^2}{2\sigma^2}.
$$

Therefore,

$$
\begin{aligned}
h(X)
&=-\mathbb E[\log p(X)]\\
&=\frac12\log(2\pi\sigma^2)
+\frac{\mathbb E[(X-\mu)^2]}{2\sigma^2}\\
&=\frac12\log(2\pi e\sigma^2).
\end{aligned}
$$


**Multivariate Gaussian Entropy.**

Let $X\sim\mathcal N_d(\mu,\Sigma)$, where
$\mu\in\mathbb R^d$ and $\Sigma$ is a positive-definite
covariance matrix.

Its probability density is

$$
p(x)=\frac{1}{(2\pi)^{d/2}(\det\Sigma)^{1/2}}
\exp\left(
-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)
\right).
$$

Taking the negative expected log-density,

$$
\begin{aligned}
h(X)
&=-\mathbb E[\log p(X)]\\
&=\frac d2\log(2\pi)
+\frac12\log\det\Sigma\\
&\quad+\frac12\mathbb E[
(X-\mu)^\top\Sigma^{-1}(X-\mu)].
\end{aligned}
$$

Using the trace identity,

$$
\begin{aligned}
&\mathbb E[
(X-\mu)^\top\Sigma^{-1}(X-\mu)]\\
&=\operatorname{tr}\left(
\Sigma^{-1}\mathbb E[
(X-\mu)(X-\mu)^\top]\right)\\
&=\operatorname{tr}(\Sigma^{-1}\Sigma)=d.
\end{aligned}
$$

Therefore,

$$
\boxed{
h(X)=\frac12\log\left((2\pi e)^d\det\Sigma\right).
}
$$

Since $\det\Sigma$ equals the product of the
eigenvalues of $\Sigma$, the entropy depends on
the overall volume scale of the Gaussian distribution.


### 3.3 Conditional Entropy

Conditional entropy measures the remaining uncertainty about $Y$ after observing $X$.

For discrete random variables, it is defined as

$$
\boxed{
H(Y\mid X)
=
-\sum_{x\in\mathcal X,\;y\in\mathcal Y}
p(x,y)\log\frac{p(x,y)}{p(x)}.
}
$$

Using $p(y\mid x)=p(x,y)/p(x)$, we can equivalently write

$$
H(Y\mid X)
=
-\mathbb E_{p(x,y)}[\log p(Y\mid X)].
$$

For continuous random variables, the conditional differential entropy is

$$
\boxed{
h(Y\mid X)
=
-\int\int p(x,y)\log p(y\mid x)\,dy\,dx.
}
$$

**Chain Rule for Entropy.**

Using $p(x,y)=p(y\mid x)p(x)$,

$$
\begin{aligned}
H(X,Y)
&=-\mathbb E_{p(x,y)}[\log p(X,Y)]\\
&=-\mathbb E_{p(x,y)}[\log p(Y\mid X)]
-\mathbb E_{p(x)}[\log p(X)]\\
&=H(Y\mid X)+H(X).
\end{aligned}
$$

Therefore,

$$
\boxed{
H(X,Y)=H(X)+H(Y\mid X).
}
$$

The same chain rule holds for differential entropy, replacing $H$ with $h$, whenever the relevant quantities are well-defined.


### 3.4 Cross-Entropy

Entropy measures uncertainty under a single distribution.
Cross-entropy measures the expected negative log-probability
assigned by a distribution $q$ to observations drawn from $p$.

For discrete distributions,

$$
H(p,q)=-\sum_x p(x)\log q(x).
$$

For continuous densities, the corresponding expression is

$$
\boxed{
H(p,q)=-\mathbb E_{X\sim p}[\log q(X)].
}
$$

When $p=q$, cross-entropy reduces to entropy.

### 3.5 Connection to Negative Log-Likelihood

Suppose observations $x_1,\ldots,x_n$ are independently drawn from an unknown data distribution $p_{\mathrm{data}}$.

We approximate this distribution using a parameterized density $p_\theta(x)$.

The likelihood of the observations is

$$
L(\theta)
=
\prod_{i=1}^{n}p_\theta(x_i).
$$

Taking logarithms gives

$$
\log L(\theta)
=
\sum_{i=1}^{n}\log p_\theta(x_i).
$$

Maximum likelihood estimation is therefore equivalent to minimizing the average negative log-likelihood:

$$
\boxed{
\mathcal L_{\mathrm{NLL}}(\theta)
=
-\frac1n\sum_{i=1}^{n}\log p_\theta(x_i).
}
$$

For a fixed $\theta$, assume the observations are i.i.d. and the expected absolute log-likelihood is finite:

$$
\mathbb E_{p_{\mathrm{data}}}[|\log p_\theta(X)|]<\infty.
$$

Then the law of large numbers gives

$$
\begin{aligned}
&-\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(X_i)\\
&\quad\xrightarrow{\mathrm{a.s.}} H(p_{\mathrm{data}},p_\theta),
\end{aligned}
$$

where $H(p_{\mathrm{data}},p_\theta)=-\mathbb E_{p_{\mathrm{data}}}[\log p_\theta(X)]$.

Thus, empirical NLL estimates the population cross-entropy.

Moreover, when the relevant quantities are finite,

$$
H(p_{\mathrm{data}},p_\theta)
=
h(p_{\mathrm{data}})
+
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta).
$$

Since $h(p_{\mathrm{data}})$ does not depend on $\theta$, minimizing population cross-entropy is equivalent to minimizing the forward KL divergence.


## 4. KL Divergence

### 4.1 Definition

The Kullback-Leibler (KL) divergence measures how one probability distribution differs from another.

For continuous distributions with densities $p$ and $q$,

$$
\boxed{
D_{\mathrm{KL}}(p\|q)
=
\int p(x)\log\frac{p(x)}{q(x)}\,dx
=
\mathbb E_p\left[\log\frac{p(X)}{q(X)}\right].
}
$$

We assume $p\ll q$, meaning that $q(x)=0$ implies $p(x)=0$ almost everywhere. Otherwise, the KL divergence
is defined as $+\infty$.

Unlike a distance metric, KL divergence is generally asymmetric:

$$
D_{\mathrm{KL}}(p\|q)
\neq
D_{\mathrm{KL}}(q\|p).
$$

### 4.2 Gibbs' Inequality

KL divergence is always non-negative:

$$
\boxed{
D_{\mathrm{KL}}(p\|q)\geq0.
}
$$

**Proof.**

Using the inequality $\log u\leq u-1$ for $u>0$,

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&=-\int_{\{p>0\}}
p(x)\log\frac{q(x)}{p(x)}\,dx\\
&\geq
\int_{\{p>0\}}
p(x)\left(1-\frac{q(x)}{p(x)}\right)dx\\
&=1-\int_{\{p>0\}}q(x)\,dx\\
&\geq0.
\end{aligned}
$$

Equality holds if and only if $p=q$ almost everywhere.

### 4.3 Forward and Reverse KL

Because KL divergence is asymmetric, the order
of the distributions matters.

Suppose $p$ is a target distribution and $q$ is
an approximation.

**Forward KL:**

<img src="/assets/images/generative-models/forwardkl.png" alt="Forward KL distribution comparison" style="width: 60%; height: auto; display: block; margin: 1em auto;" />

$$
D_{\mathrm{KL}}(p\|q)
=
\mathbb E_p\left[\log\frac{p(X)}{q(X)}\right].
$$

This penalizes $q$ for assigning low probability to regions where $p$ has substantial probability mass.


**Reverse KL:**

<img src="/assets/images/generative-models/reversekl.png" alt="Reverse KL distribution comparison" style="width: 60%; height: auto; display: block; margin: 1em auto;" />
$$
D_{\mathrm{KL}}(q\|p)
=
\mathbb E_q\left[\log\frac{q(X)}{p(X)}\right].
$$

This penalizes $q$ for placing probability mass in regions where $p$ has low probability.

Consequently, forward KL often encourages mode-covering behavior, while reverse KL can favor mode-seeking approximations. However, this behavior depends on the approximation family. If the family is flexible enough to represent the target distribution exactly, both KL objectives are minimized when $q=p$.

### 4.4 KL Divergence Between Gaussian Distributions

Consider two $d$-dimensional Gaussian distributions:

$$
p=\mathcal N(\mu_p,\Sigma_p),
\qquad
q=\mathcal N(\mu_q,\Sigma_q),
$$

where both covariance matrices are positive definite.

We want to calculate $D_{\mathrm{KL}}(p\|q)$.

From the definition,

$$
D_{\mathrm{KL}}(p\|q)
=
\mathbb E_p[\log p(X)-\log q(X)].
$$

The log-density of a multivariate Gaussian is

$$
\log p(x)
=
-\frac d2\log(2\pi)
-\frac12\log\det\Sigma_p
-\frac12(x-\mu_p)^\top
\Sigma_p^{-1}(x-\mu_p).
$$

Similarly,

$$
\log q(x)
=
-\frac d2\log(2\pi)
-\frac12\log\det\Sigma_q
-\frac12(x-\mu_q)^\top
\Sigma_q^{-1}(x-\mu_q).
$$

Subtracting and taking expectations under $p$,

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
=\frac12\Big[
&\log\frac{\det\Sigma_q}{\det\Sigma_p}+\mathbb E_p[
(X-\mu_q)^\top\Sigma_q^{-1}(X-\mu_q)]-\mathbb E_p[
(X-\mu_p)^\top\Sigma_p^{-1}(X-\mu_p)]
\Big].
\end{aligned}
$$

The final expectation equals $d$:

$$
\mathbb E_p[
(X-\mu_p)^\top\Sigma_p^{-1}(X-\mu_p)]
=
d.
$$

For the other quadratic term, write

$$
X-\mu_q=(X-\mu_p)+(\mu_p-\mu_q).
$$

Since $\mathbb E_p[X-\mu_p]=0$,

$$
\begin{aligned}
&\mathbb E_p[
(X-\mu_q)^\top\Sigma_q^{-1}(X-\mu_q)]\\
&=\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)\\
&\quad+
(\mu_p-\mu_q)^\top
\Sigma_q^{-1}(\mu_p-\mu_q).
\end{aligned}
$$

Combining these terms gives

$$
\boxed{
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
=\frac12\Big[
&\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)+(\mu_p-\mu_q)^\top
\Sigma_q^{-1}(\mu_p-\mu_q) -d+\log\frac{\det\Sigma_q}{\det\Sigma_p}
\Big].
\end{aligned}
}
$$

### 4.5 Special Case: Diagonal Gaussian vs Standard Normal

Let

$$
p=\mathcal N(\mu,\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2)),
\qquad
q=\mathcal N(0,I).
$$

Substituting into the Gaussian KL formula,

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&=\frac12\left[
\operatorname{tr}(\Sigma)
+\mu^\top\mu-d-\log\det\Sigma
\right]\\
&=\frac12\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right).
\end{aligned}
$$

Therefore,

$$
\boxed{
D_{\mathrm{KL}}(p\|\mathcal N(0,I))
=
\frac12\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right).
}
$$

This closed-form expression is useful whenever a diagonal Gaussian is compared with a standard normal distribution.


## 5. Mutual Information

Mutual information measures how much information two random variables share. It is zero if and only if they are independent.

### 5.1 Definition

Mutual information is defined as the KL divergence between the joint distribution and the product of the marginal distributions:

$$
\boxed{
I(X;Z)
=
D_{\mathrm{KL}}(p(x,z)\|p(x)p(z)).
}
$$

For continuous random variables,

$$
I(X;Z)
=
\int p(x,z)
\log\frac{p(x,z)}{p(x)p(z)}
\,dx\,dz.
$$

If $X$ and $Z$ are independent, then $p(x,z)=p(x)p(z)$, so $I(X;Z)=0$.

### 5.2 Equivalent Representations

For continuous random variables, mutual information can be expressed in terms of differential entropy.

Using $p(x,z)=p(x\mid z)p(z)$,

$$
\begin{aligned}
I(X;Z)
&=\mathbb E_{p(x,z)}
\left[\log\frac{p(x\mid z)}{p(x)}\right]\\
&=\mathbb E_{p(x,z)}[\log p(x\mid z)]
-\mathbb E_{p(x)}[\log p(x)]\\
&=h(X)-h(X\mid Z).
\end{aligned}
$$

The last equality follows from the definitions

$$
h(X)=-\mathbb E_{p(x)}[\log p(x)]
$$

and

$$
h(X\mid Z)=-\mathbb E_{p(x,z)}[\log p(x\mid z)].
$$

Similarly, exchanging $X$ and $Z$ gives

$$
\boxed{
I(X;Z)
=
h(X)-h(X\mid Z)
=
h(Z)-h(Z\mid X).
}
$$

Thus, mutual information measures the reduction in uncertainty about one variable after observing the other.

For discrete random variables, the same identities hold with Shannon entropy $H$ replacing differential entropy $h$.

**Conditional KL representation.**

Using the definition of conditional probability,

$$
\begin{aligned}
I(X;Z)
&=
\int p(x,z)
\log\frac{p(z\mid x)}{p(z)}
\,dx\,dz\\
&=
\int p(x)
\left[
\int p(z\mid x)
\log\frac{p(z\mid x)}{p(z)}
\,dz
\right]dx.
\end{aligned}
$$

Therefore,

$$
\boxed{
I(X;Z)
=
\mathbb E_{p(x)}
\left[
D_{\mathrm{KL}}(p(z\mid x)\|p(z))
\right].
}
$$

This shows that mutual information measures how much observing $X$ changes our distribution over $Z$, on average.


## 6. Change of Variables and Pushforward Distributions

### 6.1 Pushforward Distribution

Let $Z$ be a random variable with distribution $P_Z$, and define a new random variable

$$
X=f(Z).
$$

The distribution of $X$ is called the pushforward of $P_Z$ through $f$, written as

$$
\boxed{
P_X=f_\#P_Z.
}
$$

By definition, for any measurable set $A$,

$$
P_X(A)
=
P_Z(f^{-1}(A)).
$$

In other words, the probability that $X$ belongs to $A$ equals the probability that $Z$ belongs to the preimage of $A$.

**Interpretation.** A pushforward distribution describes how a probability distribution changes under a transformation $f$. The values and density may change, but the probability of corresponding events is preserved.

<img src="/assets/images/generative-models/image-20261014020138037.png" alt="Pushforward probability distribution" style="width: 60%; height: auto; display: block; margin: 1em auto;" />

​						*Figure 1. Pushforward of a probability distribution under a mapping.*


### 6.2 One-Dimensional Change of Variables

Suppose $f:\mathbb R\to\mathbb R$ is differentiable, invertible, and has a nonzero derivative.

Since $X=f(Z)$,

$$
P(X\leq x)
=
P(Z\leq f^{-1}(x))
$$

when $f$ is strictly increasing.

Differentiating the CDF gives

$$
p_X(x)
=
p_Z(f^{-1}(x))
\left|
\frac{d}{dx}f^{-1}(x)
\right|.
$$

Equivalently, for $x=f(z)$,

$$
\boxed{
p_X(f(z))
=
\frac{p_Z(z)}{|f'(z)|}.
}
$$

The derivative accounts for how the transformation stretches or compresses the space.

### 6.3 Multivariate Change of Variables

Now let $f:\mathbb R^d\to\mathbb R^d$ be an invertible differentiable transformation with a nonsingular Jacobian.

The Jacobian matrix is

$$
J_f(z)
=
\frac{\partial f(z)}{\partial z}.
$$

Its determinant measures the local volume change under the transformation.

The density transformation formula is

$$
\boxed{
p_X(x)
=
p_Z(f^{-1}(x))
\left|
\det J_{f^{-1}}(x)
\right|.
}
$$

Equivalently, at $x=f(z)$,

$$
p_X(f(z))
=
\frac{p_Z(z)}
{|\det J_f(z)|}.
$$

Taking logarithms,

$$
\boxed{
\log p_X(f(z))
=
\log p_Z(z)
-
\log|\det J_f(z)|.
}
$$

This formula allows us to calculate the density of a transformed random variable from its original density and the Jacobian determinant.

For non-invertible transformations, the pushforward distribution is still well-defined, but the simple determinant formula generally requires modification.


## 7. Summary and Connections

| Mathematical idea | Later appearance |
|:--|:--|
| Marginalization $p(x)=\int p(x\mid z)p(z)dz$ | Latent-variable models, VAE |
| Chain rule for joint distributions | Autoregressive generation |
| Conditional expectation as MSE minimizer | Denoising, conditional flow matching |
| Negative log-likelihood / cross-entropy | Maximum-likelihood training |
| Gaussian KL | VAE regularization, diffusion ELBO |
| Mutual information | Representation learning, information bottleneck |
| Change of variables | Normalizing flows, continuous flows |

