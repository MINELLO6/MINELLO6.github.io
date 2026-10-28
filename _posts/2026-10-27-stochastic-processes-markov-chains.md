---
layout: post
title: "M06 - Stochastic Processes and Markov Chains"
date: 2026-10-27
mathjax: true
comments: true
---

# Stochastic Processes and Markov Chains

**Prerequisites:** M01 — Probability & Information Theory; M04 — Gaussian Identities & Matrix Calculus.

**Aim:** Understand discrete-time stochastic processes, Markov chains, transition kernels, and time reversal. Use these concepts to derive the DDPM forward diffusion process, and establish the foundations for Brownian motion and stochastic differential equations.

---

## 1. Why Stochastic Processes?

In previous modules, we primarily studied individual random variables and probability distributions.

A stochastic process extends this idea by considering a collection of random variables indexed by time.

For example,

$$
X_0,X_1,\ldots,X_T
$$

represents the evolution of a random system over discrete time.

Each $X_t$ is a random variable, and the variables may be statistically dependent.

In generative modeling, stochastic processes describe how probability distributions evolve.

For example, diffusion models gradually transform data into noise:

$$
x_0\longrightarrow x_1\longrightarrow\cdots
\longrightarrow x_T.
$$

A learned reverse process attempts to transform noise back into data:

$$
x_T\longrightarrow x_{T-1}\longrightarrow\cdots
\longrightarrow x_0.
$$

Understanding these transformations requires the mathematical language of stochastic processes and Markov chains.

---

## 2. Definition of a Stochastic Process

### 2.1 Mathematical Definition

A stochastic process is a collection of random variables

$$
\boxed{
\{X_t:t\in\mathcal T\}
}
$$

defined on a common probability space.

The index set $\mathcal T$ is usually interpreted as time.

Examples include:

- Discrete time: $\mathcal T=\{0,1,2,\ldots\}$.
- Continuous time: $\mathcal T=[0,\infty)$.

The state space describes the possible values of $X_t$.

For example:

$$
X_t\in\{0,1\}
$$

defines a binary-state process, while

$$
X_t\in\mathbb R^d
$$

defines a continuous-state process.

DDPM uses a discrete-time stochastic process with continuous-valued states.

### 2.2 Random Variables vs Sample Paths

A stochastic process can be viewed in two complementary ways.

**Fixed time**

For a fixed $t$, $X_t$ is a random variable with some distribution

$$
p_t(x).
$$

**Fixed outcome**

For a particular outcome $\omega$ in the underlying probability space, we obtain a trajectory

$$
t\mapsto X_t(\omega).
$$

This is called a sample path.

Thus:

- A distribution describes randomness across possible states at a particular time.
- A sample path describes one possible evolution over time.

These are different objects.

A diffusion trajectory represents one realization of a stochastic process, while the marginal distribution describes all possible states at that time.

### 2.3 Finite-Dimensional Distributions

A stochastic process is characterized by its joint distributions at collections of time points.

For example,

$$
p(x_0,x_1,\ldots,x_T)
$$

describes the joint behavior of the entire discrete-time trajectory.

Marginal distributions are obtained by integrating out other variables:

$$
p_t(x_t)
=
\int
p(x_0,\ldots,x_T)
\prod_{s\neq t}dx_s.
$$

The central question is how to describe this high-dimensional joint distribution efficiently.

Markov processes provide one answer.

---

## 3. The Markov Property

### 3.1 Definition

A discrete-time process is Markov if the conditional distribution of its next state depends on the past only through the current state.

Formally,

$$
\boxed{
P(X_t\in A\mid X_0,\ldots,X_{t-1})
=
P(X_t\in A\mid X_{t-1})
}
$$

for measurable sets $A$, whenever the conditional probabilities are defined.

When transition densities exist, we write

$$
\boxed{
p(x_t\mid x_{t-1},\ldots,x_0)
=
p(x_t\mid x_{t-1}).
}
$$

This is the Markov property.

### 3.2 Intuition

Consider a person walking randomly along a straight line.

At each time step, the person moves left or right.

If the next movement depends only on the current position, the position process is Markov.

Knowing the complete history of previous positions does not provide additional predictive information once the current position is known.

However, if the movement depends on the previous direction, position alone may not satisfy the Markov property.

One solution is to expand the state to include both position and direction.

This illustrates an important principle:

**Whether a process is Markov depends on how its state is defined.**

### 3.3 Markov Does Not Mean Independent

The Markov property does not imply that

$$
X_t\perp X_{t-1}.
$$

In fact, adjacent states are usually highly dependent.

Instead, the relevant conditional independence is

$$
\boxed{
X_t
\perp
(X_0,\ldots,X_{t-2})
\mid X_{t-1}.
}
$$

The current state contains the information from the past required to predict the next state.

---

## 4. Factorization of a Markov Chain

For an arbitrary stochastic process, the chain rule gives

$$
\begin{aligned}
p(x_0,\ldots,x_T)
={}&p(x_0)\\
&\times p(x_1\mid x_0)\\
&\times p(x_2\mid x_1,x_0)\\
&\times\cdots\\
&\times p(x_T\mid x_{T-1},\ldots,x_0).
\end{aligned}
$$

Under the Markov property,

$$
p(x_t\mid x_{t-1},\ldots,x_0)
=
p(x_t\mid x_{t-1}).
$$

Therefore,

$$
\boxed{
p(x_0,\ldots,x_T)
=
p(x_0)
\prod_{t=1}^{T}
p(x_t\mid x_{t-1}).
}
$$

This factorization is fundamental.

Instead of directly specifying a potentially complicated joint distribution, we specify:

1. An initial distribution $p(x_0)$.
2. A collection of transition distributions $p(x_t\mid x_{t-1})$.

### 4.1 Connection to Diffusion Models

The DDPM forward process is written as

$$
\boxed{
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
}
$$

Here,

$$
x_{1:T}=(x_1,\ldots,x_T).
$$

The distribution $q(x_t\mid x_{t-1})$ describes one forward noising step.

The full conditional distribution $q(x_{1:T}\mid x_0)$ describes all noisy states given the original data sample.

This is simply the Markov factorization applied to the forward diffusion process.

---

## 5. Transition Kernels

### 5.1 Definition

A transition kernel describes how a stochastic process moves from one state to another.

For a Markov process, define

$$
K_t(x,A)
=
P(X_t\in A\mid X_{t-1}=x).
$$

For each fixed $x$, $K_t(x,\cdot)$ is a probability measure.

For each measurable set $A$, $K_t(\cdot,A)$ is a measurable function.

If a transition density exists,

$$
K_t(x,A)
=
\int_A p_t(y\mid x)\,dy.
$$

The distinction is important because not all Markov processes have transition densities.

Transition kernels are the more general mathematical object.

### 5.2 Time-Homogeneous and Time-Inhomogeneous Processes

A Markov chain is time-homogeneous if its transition kernel does not depend on time:

$$
K_t(x,A)=K(x,A).
$$

Otherwise, the chain is time-inhomogeneous.

DDPM is generally time-inhomogeneous because its noise level changes with $t$.

For example,

$$
q(x_t\mid x_{t-1})
=
\mathcal N(
\sqrt{\alpha_t}x_{t-1},
\beta_tI
),
$$

where $\alpha_t$ and $\beta_t$ depend on the time step.

### 5.3 Discrete-State Transition Matrices

For a finite state space, transition probabilities can be represented using a matrix.

Consider two states:

$$
\mathcal S=\{0,1\}.
$$

Define

$$
P=
\begin{pmatrix}
0.9&0.1\\
0.2&0.8
\end{pmatrix}.
$$

Using the row-vector convention,

$$
P_{ij}
=
P(X_{t+1}=j\mid X_t=i).
$$

Thus:

- From state 0, there is a 90% chance of remaining in state 0.
- From state 0, there is a 10% chance of moving to state 1.
- From state 1, there is a 20% chance of moving to state 0.
- From state 1, there is an 80% chance of remaining in state 1.

Every row sums to one.

---

## 6. Chapman–Kolmogorov Equation

The Chapman–Kolmogorov equation describes how to compose transitions over multiple time steps.

### 6.1 Discrete-State Version

Suppose we want

$$
P(X_{t+2}=j\mid X_t=i).
$$

We sum over every possible intermediate state $k$:

$$
\begin{aligned}
&P(X_{t+2}=j\mid X_t=i)\\
&=
\sum_k
P(X_{t+2}=j\mid X_{t+1}=k)
P(X_{t+1}=k\mid X_t=i).
\end{aligned}
$$

For a time-homogeneous Markov chain,

$$
\boxed{
P^{(2)}=P^2.
}
$$

More generally,

$$
\boxed{
P^{(n)}=P^n.
}
$$

Here, $P^{(n)}$ denotes the $n$-step transition matrix.

The matrix power appears because we sum over all intermediate states.

### 6.2 Continuous-State Version

For a Markov process with transition densities,

$$
\boxed{
p(x_t\mid x_s)
=
\int
p(x_t\mid x_u)
p(x_u\mid x_s)
\,dx_u,
}
$$

where

$$
s<u<t.
$$

This equation describes the composition of two probabilistic transitions.

### 6.3 Marginal Distribution Propagation

Suppose the marginal density at time $t-1$ is

$$
p_{t-1}(x).
$$

Then

$$
\boxed{
p_t(y)
=
\int
p_t(y\mid x)
p_{t-1}(x)
\,dx.
}
$$

In the finite-state case, using row vectors,

$$
\boxed{
\pi_t=\pi_{t-1}P_t.
}
$$

This is a fundamental relationship between transition probabilities and evolving marginal distributions.

In diffusion models, the transition distribution gradually modifies the data distribution.

---

## 7. Stationary Distributions

A stationary distribution is a probability distribution that remains unchanged under a transition kernel.

For a time-homogeneous finite-state Markov chain,

$$
\boxed{
\pi=\pi P.
}
$$

For the matrix

$$
P=
\begin{pmatrix}
0.9&0.1\\
0.2&0.8
\end{pmatrix},
$$

let

$$
\pi=(\pi_0,\pi_1).
$$

The stationary equations are

$$
\pi_0=0.9\pi_0+0.2\pi_1
$$

and

$$
\pi_0+\pi_1=1.
$$

Rearranging,

$$
0.1\pi_0=0.2\pi_1.
$$

Therefore,

$$
\boxed{
\pi=
\left(
\frac23,\frac13
\right).
}
$$

If the process starts from this distribution, it retains the same marginal distribution at every time step.

For this finite irreducible and aperiodic chain, the distribution also converges to $\pi$ from any initial distribution.

Not every Markov chain has a unique stationary distribution, and convergence is not guaranteed without appropriate assumptions.

### 7.1 Stationarity vs Time-Homogeneity

These concepts are different.

- Time-homogeneity means the transition rule does not change with time.
- Stationarity means the process has time-invariant finite-dimensional distributions.

A time-homogeneous Markov chain need not start in its stationary distribution.

The standard DDPM forward process is time-inhomogeneous, and its marginals generally change over time.

Its terminal distribution is designed to be approximately a simple Gaussian prior.

This does not mean every intermediate distribution is stationary.

---

## 8. Time Reversal of a Markov Chain

This section is essential for understanding generative diffusion models.

### 8.1 Forward and Reverse Conditional Distributions

Suppose the forward transition is

$$
p(x_t\mid x_{t-1}).
$$

The reverse conditional distribution is

$$
p(x_{t-1}\mid x_t).
$$

Using Bayes' rule,

$$
\boxed{
p(x_{t-1}\mid x_t)
=
\frac{
p(x_t\mid x_{t-1})p_{t-1}(x_{t-1})
}{
p_t(x_t)
}.
}
$$

The reverse transition requires the marginal distributions at the relevant times.

It cannot generally be obtained by simply changing the direction of the forward transition.

### 8.2 Why Reverse Transitions Are Difficult

The forward transition may be simple.

For example,

$$
p(x_t\mid x_{t-1})
=
\mathcal N(
x_t;\sqrt{\alpha_t}x_{t-1},
\beta_tI
).
$$

However, the reverse conditional distribution depends on

$$
p_{t-1}(x_{t-1})
$$

and

$$
p_t(x_t).
$$

When the initial distribution is a complicated high-dimensional data distribution, these marginals are generally difficult to evaluate.

Consequently,

$$
p(x_{t-1}\mid x_t)
$$

may be complicated even when the forward transitions are Gaussian.

This is a central reason diffusion models use neural networks to approximate reverse transitions.

### 8.3 A Discrete-State Example

For a finite Markov chain, let

$$
P_{ij}=P(X_{t+1}=j\mid X_t=i).
$$

The reverse transition probability is

$$
\boxed{
R_t(j,i)
=
\frac{
\pi_t(i)P_{ij}
}{
\pi_{t+1}(j)
},
}
$$

provided $\pi_{t+1}(j)>0$.

The reverse matrix generally depends on the time-dependent marginal distributions.

In a stationary chain, it becomes

$$
R(j,i)
=
\frac{\pi(i)P_{ij}}{\pi(j)}.
$$

A stationary chain is called reversible if

$$
\boxed{
\pi(i)P_{ij}
=
\pi(j)P_{ji}.
}
$$

This is the detailed balance condition.

Time reversal exists more generally; detailed balance is a special property, not a requirement for reversing a Markov process.

---

## 9. Gaussian Markov Chains

We now combine Markov chains with the Gaussian identities from M04.

### 9.1 Linear Gaussian Transition

Consider

$$
\boxed{
X_t=A_tX_{t-1}+b_t+\epsilon_t
}
$$

where

$$
\epsilon_t\sim\mathcal N(0,Q_t)
$$

and the innovations $\epsilon_t$ are independent of the previous states and each other.

The conditional distribution is

$$
\boxed{
X_t\mid X_{t-1}=x
\sim
\mathcal N(A_tx+b_t,Q_t).
}
$$

This defines a Gaussian Markov chain.

### 9.2 Propagating a Gaussian Distribution

Suppose

$$
X_{t-1}\sim\mathcal N(m_{t-1},P_{t-1}).
$$

From the affine Gaussian identity,

$$
\boxed{
m_t=A_tm_{t-1}+b_t
}
$$

and

$$
\boxed{
P_t=A_tP_{t-1}A_t^\top+Q_t.
}
$$

Therefore,

$$
X_t\sim\mathcal N(m_t,P_t).
$$

These equations appear in linear Gaussian state-space models and Kalman filtering.

They are also closely related to the DDPM forward process.

### 9.3 The DDPM Special Case

Set

$$
A_t=\sqrt{\alpha_t}I,
$$

$$
b_t=0,
$$

and

$$
Q_t=\beta_tI.
$$

Then

$$
\boxed{
X_t
=
\sqrt{\alpha_t}X_{t-1}
+
\sqrt{\beta_t}\epsilon_t,
}
$$

where

$$
\epsilon_t\sim\mathcal N(0,I)
$$

and

$$
\alpha_t=1-\beta_t.
$$

This is precisely the Gaussian noising transition used in DDPM.

---

## 10. Deriving the DDPM Forward Process

This is the most important derivation in M05.

### 10.1 One-Step Forward Distribution

The forward transition is

$$
\boxed{
q(x_t\mid x_{t-1})
=
\mathcal N(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_tI
).
}
$$

Equivalently,

$$
\boxed{
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon_t,
}
$$

where

$$
\epsilon_t\sim\mathcal N(0,I).
$$

The noise variables are assumed independent across time steps.

The clean image $x_0$ is treated as fixed when deriving the conditional distribution $q(x_t\mid x_0)$.

### 10.2 Two Forward Steps

For the first step,

$$
x_1
=
\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}\epsilon_1.
$$

For the second step,

$$
x_2
=
\sqrt{\alpha_2}x_1
+
\sqrt{\beta_2}\epsilon_2.
$$

Substituting $x_1$,

$$
\begin{aligned}
x_2
={}&\sqrt{\alpha_2}
\left(
\sqrt{\alpha_1}x_0+
\sqrt{\beta_1}\epsilon_1
\right)\\
&+\sqrt{\beta_2}\epsilon_2.
\end{aligned}
$$

Therefore,

$$
x_2
=
\sqrt{\alpha_1\alpha_2}x_0
+
\sqrt{\alpha_2\beta_1}\epsilon_1
+
\sqrt{\beta_2}\epsilon_2.
$$

Since the noise variables are independent Gaussians, their linear combination is Gaussian.

The conditional mean is

$$
\mathbb E[x_2\mid x_0]
=
\sqrt{\alpha_1\alpha_2}x_0.
$$

The conditional covariance is

$$
\operatorname{Var}(x_2\mid x_0)
=
(\alpha_2\beta_1+\beta_2)I.
$$

Recall

$$
\beta_t=1-\alpha_t.
$$

Then

$$
\begin{aligned}
\alpha_2\beta_1+\beta_2
&=\alpha_2(1-\alpha_1)+(1-\alpha_2)\\
&=1-\alpha_1\alpha_2.
\end{aligned}
$$

Thus,

$$
\boxed{
q(x_2\mid x_0)
=
\mathcal N(
\sqrt{\alpha_1\alpha_2}x_0,
(1-\alpha_1\alpha_2)I
).
}
$$

### 10.3 General Result

Define

$$
\boxed{
\bar\alpha_t
=
\prod_{s=1}^{t}\alpha_s.
}
$$

We claim that

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
).
}
$$

We prove this by induction.

Assume

$$
x_{t-1}\mid x_0
\sim
\mathcal N(
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
).
$$

The next transition is

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon_t.
$$

Its conditional mean is

$$
\begin{aligned}
\mathbb E[x_t\mid x_0]
&=\sqrt{\alpha_t}
\mathbb E[x_{t-1}\mid x_0]\\
&=\sqrt{\alpha_t}
\sqrt{\bar\alpha_{t-1}}x_0\\
&=\sqrt{\bar\alpha_t}x_0.
\end{aligned}
$$

Its conditional covariance is

$$
\begin{aligned}
\operatorname{Var}(x_t\mid x_0)
&=
\alpha_t(1-\bar\alpha_{t-1})I
+\beta_tI\\
&=
(\alpha_t-\alpha_t\bar\alpha_{t-1}
+1-\alpha_t)I\\
&=
(1-\bar\alpha_t)I.
\end{aligned}
$$

Therefore,

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
).
}
$$

This completes the induction.

### 10.4 Direct Sampling Formula

Using Gaussian reparameterization,

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

This formula allows us to sample $x_t$ directly from $x_0$.

We do not need to generate

$$
x_1,x_2,\ldots,x_{t-1}
$$

one by one.

This is extremely useful during DDPM training, where we can randomly select a time step $t$ and generate its noisy input directly.

---

## 11. What Happens as Time Increases?

Recall

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

The two terms have different interpretations.

The first term,

$$
\sqrt{\bar\alpha_t}x_0,
$$

represents the remaining signal from the original data.

The second term,

$$
\sqrt{1-\bar\alpha_t}\epsilon,
$$

represents accumulated Gaussian noise.

### 11.1 Small Noise

When $t$ is small,

$$
\bar\alpha_t\approx1.
$$

Then

$$
x_t\approx x_0.
$$

The noisy observation still contains most of the original information.

### 11.2 Large Noise

When $t$ becomes sufficiently large and

$$
\bar\alpha_t\approx0,
$$

we obtain

$$
x_t\approx\epsilon.
$$

Therefore,

$$
q(x_t\mid x_0)\approx\mathcal N(0,I).
$$

If the signal coefficient becomes negligible and the data distribution satisfies suitable regularity conditions, the marginal distribution $q(x_t)$ is also close to a standard Gaussian.

The exact degree of approximation depends on the noise schedule and the number of steps.

### 11.3 Signal-to-Noise Ratio

Conditioning on a fixed $x_0$, the signal coefficient squared is $\bar\alpha_t$, while the noise variance per dimension is $1-\bar\alpha_t$.

This motivates the conventional diffusion signal-to-noise ratio:

$$
\boxed{
\operatorname{SNR}(t)
=
\frac{\bar\alpha_t}{1-\bar\alpha_t}.
}
$$

Its logarithm is

$$
\boxed{
\operatorname{logSNR}(t)
=
\log\bar\alpha_t
-
\log(1-\bar\alpha_t).
}
$$

At early time steps, the SNR is typically high.

At late time steps, the SNR becomes small.

The SNR describes the relative scale of the signal and noise coefficients. It is not necessarily the empirical signal-to-noise power ratio of a particular dataset.

---

## 12. The Exact Gaussian Forward Posterior

We now connect the forward Markov process with the Gaussian product identity from M04.

The quantity of interest is

$$
\boxed{
q(x_{t-1}\mid x_t,x_0).
}
$$

Unlike the generally complicated reverse conditional $q(x_{t-1}\mid x_t)$, this distribution has an analytical Gaussian form.

### 12.1 Using Bayes' Rule

From the Markov property,

$$
q(x_t\mid x_{t-1},x_0)
=
q(x_t\mid x_{t-1}).
$$

Therefore,

$$
\begin{aligned}
q(x_{t-1}\mid x_t,x_0)
&=
\frac{
q(x_t\mid x_{t-1},x_0)
q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}\\
&\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0).
\end{aligned}
$$

We already know both factors.

The first is

$$
q(x_t\mid x_{t-1})
=
\mathcal N(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_tI
).
$$

The second is

$$
q(x_{t-1}\mid x_0)
=
\mathcal N(
x_{t-1};
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
).
$$

We multiply these two Gaussian factors as functions of $x_{t-1}$.

### 12.2 Completing the Square

Ignoring terms independent of $x_{t-1}$,

$$
\begin{aligned}
&\log q(x_{t-1}\mid x_t,x_0)\\
&=
-\frac{
\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
}{
2\beta_t
}\\
&\quad-
\frac{
\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\|^2
}{
2(1-\bar\alpha_{t-1})
}
+\text{constant}.
\end{aligned}
$$

Collecting the quadratic terms in $x_{t-1}$, the posterior precision is

$$
\frac1{\tilde\beta_t}
=
\frac{\alpha_t}{\beta_t}
+
\frac1{1-\bar\alpha_{t-1}}.
$$

Simplifying,

$$
\begin{aligned}
\frac1{\tilde\beta_t}
&=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}\\
&=
\frac{
1-\bar\alpha_t
}{
\beta_t(1-\bar\alpha_{t-1})
}.
\end{aligned}
$$

Therefore,

$$
\boxed{
\tilde\beta_t
=
\frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t.
}
$$

The linear term gives

$$
\tilde\mu_t
=
\tilde\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{
\sqrt{\bar\alpha_{t-1}}
}{
1-\bar\alpha_{t-1}
}x_0
\right).
$$

Substituting the expression for $\tilde\beta_t$,

$$
\boxed{
\begin{aligned}
\tilde\mu_t(x_t,x_0)
={}&
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}x_0\\
&+
\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}x_t.
\end{aligned}
}
$$

Thus, for $t\geq2$,

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N(
x_{t-1};
\tilde\mu_t(x_t,x_0),
\tilde\beta_tI
).
}
$$

At $t=1$, conditioning on $x_0$ makes $x_{t-1}=x_0$ deterministic, so the posterior is a point mass rather than a non-degenerate Gaussian.

### 12.3 Why This Is Important

We have derived an exact conditional distribution using only Gaussian identities and the Markov property.

However, sampling from the generative reverse process does not provide access to the unknown clean sample $x_0$.

Therefore, we cannot directly use

$$
q(x_{t-1}\mid x_t,x_0)
$$

to generate new data starting from pure noise.

Instead, DDPM introduces a learned reverse model

$$
\boxed{
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
).
}
$$

The exact Gaussian posterior is used in the construction of the training objective, while the learned reverse model is used for generation.

One important distinction is:

$$
q(x_{t-1}\mid x_t,x_0)
$$

is Gaussian, but

$$
q(x_{t-1}\mid x_t)
$$

need not be Gaussian.

The latter marginalizes over the unknown original data and may be a complicated Gaussian mixture.

---

## 13. Connection to Variational Inference

The forward diffusion process defines a conditional distribution over latent variables:

$$
q(x_{1:T}\mid x_0).
$$

The reverse generative model defines

$$
\boxed{
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
}
$$

Here, the prior is typically chosen as

$$
p(x_T)=\mathcal N(0,I).
$$

The observed-data likelihood is

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}.
$$

As in variational inference, we can introduce a tractable distribution over latent variables and derive an ELBO.

The diffusion ELBO is

$$
\boxed{
\mathcal L_{\mathrm{diffusion}}
=
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\log p_\theta(x_{0:T})
-
\log q(x_{1:T}\mid x_0)
\right].
}
$$

This satisfies

$$
\mathcal L_{\mathrm{diffusion}}
\leq
\log p_\theta(x_0).
$$

The mathematical structure resembles the VAE ELBO from M03.

The difference is that DDPM introduces a sequence of latent states rather than a single latent variable.

We will derive the full decomposition into Gaussian KL terms in the DDPM module.

---

## 14. From Random Walks to Brownian Motion

So far, we have considered discrete-time stochastic processes.

We now introduce the idea that motivates continuous-time stochastic processes.

### 14.1 Gaussian Random Walk

Consider

$$
X_{k+1}
=
X_k+\sigma\epsilon_k,
$$

where

$$
\epsilon_k\sim\mathcal N(0,1)
$$

independently.

Starting from $X_0=0$,

$$
X_n
=
\sigma\sum_{k=0}^{n-1}\epsilon_k.
$$

Therefore,

$$
\mathbb E[X_n]=0
$$

and

$$
\operatorname{Var}(X_n)=n\sigma^2.
$$

The variance increases linearly with the number of steps.

### 14.2 Time-Step Scaling

Suppose each time step has duration $\Delta t$.

A natural continuous-time scaling is

$$
\boxed{
X_{k+1}
=
X_k+\sqrt{\Delta t}\,\epsilon_k.
}
$$

After $n$ steps,

$$
X_n
=
\sum_{k=0}^{n-1}
\sqrt{\Delta t}\,\epsilon_k.
$$

Hence,

$$
\operatorname{Var}(X_n)
=
n\Delta t.
$$

If

$$
t=n\Delta t,
$$

then

$$
\boxed{
\operatorname{Var}(X_n)=t.
}
$$

This explains the square-root scaling of stochastic increments.

Using $\Delta t$ instead of $\sqrt{\Delta t}$ would give a variance of order $t\Delta t$, which vanishes as $\Delta t\to0$.

### 14.3 Brownian Motion

Standard Brownian motion $\{W_t\}_{t\geq0}$ is a continuous-time stochastic process satisfying:

1. $W_0=0$ almost surely.
2. It has independent increments.
3. For $0\leq s<t$,

   $$
   W_t-W_s\sim\mathcal N(0,t-s).
   $$

4. Its sample paths are almost surely continuous.

In particular,

$$
\boxed{
W_t\sim\mathcal N(0,t).
}
$$

Brownian motion is a fundamental building block of stochastic differential equations.

The scaled Gaussian random walk provides a discrete approximation to Brownian motion.

The precise convergence can be formalized through functional limit theorems.

### 14.4 Connection to SDEs

A stochastic differential equation is often written as

$$
\boxed{
dX_t=f(X_t,t)\,dt+g(X_t,t)\,dW_t.
}
$$

Here:

- $f$ is the drift coefficient.
- $g$ is the diffusion coefficient.
- $W_t$ is Brownian motion.

The corresponding Euler–Maruyama discretization is

$$
\boxed{
X_{t+\Delta t}
\approx
X_t+
f(X_t,t)\Delta t+
g(X_t,t)\sqrt{\Delta t}\epsilon.
}
$$

The stochastic term scales with $\sqrt{\Delta t}$ rather than $\Delta t$.

This is a consequence of the variance structure of Brownian increments.

We will study Brownian motion, Itô calculus, and stochastic differential equations more rigorously in M06 of the original mathematical foundations sequence.

---

## 15. Numerical Experiments

### Experiment 1: A Discrete Markov Chain

We simulate a two-state Markov chain and compare its evolving distribution with the stationary distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0.9, 0.1],
    [0.2, 0.8]
])

pi = np.array([1.0, 0.0])

history = [pi.copy()]

for t in range(50):
    pi = pi @ P
    history.append(pi.copy())

history = np.array(history)

stationary = np.array([2/3, 1/3])

plt.plot(history[:, 0], label="P(X_t = 0)")
plt.plot(history[:, 1], label="P(X_t = 1)")

plt.axhline(
    stationary[0],
    linestyle="--",
    label="Stationary P(0)"
)

plt.axhline(
    stationary[1],
    linestyle="--",
    label="Stationary P(1)"
)

plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Markov Chain Distribution Evolution")
plt.legend()
plt.show()
```

Observe that the distribution converges toward

$$
\pi=
\left(\frac23,\frac13\right).
$$

The transition matrix determines how the distribution evolves.

### Experiment 2: Verify the DDPM Forward Formula

We compare sequential noising with direct sampling from $q(x_T\mid x_0)$.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

T = 100
n = 20_000
x0 = 2.0

betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# Method 1: sequential noising
x_sequential = np.full(n, x0)

for t in range(T):
    eps = rng.normal(size=n)

    x_sequential = (
        np.sqrt(alphas[t]) * x_sequential
        + np.sqrt(betas[t]) * eps
    )

# Method 2: direct sampling
eps = rng.normal(size=n)

x_direct = (
    np.sqrt(alpha_bars[-1]) * x0
    + np.sqrt(1 - alpha_bars[-1]) * eps
)

print("Sequential mean:", x_sequential.mean())
print("Direct mean:", x_direct.mean())

print("Sequential variance:", x_sequential.var())
print("Direct variance:", x_direct.var())

theoretical_mean = np.sqrt(alpha_bars[-1]) * x0
theoretical_var = 1 - alpha_bars[-1]

print("Theoretical mean:", theoretical_mean)
print("Theoretical variance:", theoretical_var)

plt.hist(
    x_sequential,
    bins=60,
    density=True,
    alpha=0.5,
    label="Sequential"
)

plt.hist(
    x_direct,
    bins=60,
    density=True,
    alpha=0.5,
    label="Direct"
)

plt.legend()
plt.title("Sequential vs Direct Gaussian Noising")
plt.show()
```

The two sampling procedures should produce approximately the same distribution.

They do not produce identical individual samples because they use different random noise realizations.

This experiment verifies

$$
q(x_T\mid x_0)
=
\mathcal N(
\sqrt{\bar\alpha_T}x_0,
(1-\bar\alpha_T)I
).
$$

### Experiment 3: Visualizing the Noise Schedule

```python
import numpy as np
import matplotlib.pyplot as plt

T = 1000

betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

signal = np.sqrt(alpha_bars)
noise = np.sqrt(1 - alpha_bars)

plt.plot(signal, label="Signal coefficient")
plt.plot(noise, label="Noise coefficient")

plt.xlabel("Diffusion step t")
plt.ylabel("Coefficient")
plt.title("DDPM Forward Noise Schedule")
plt.legend()
plt.show()
```

As time increases, the coefficient multiplying the original data decreases, while the noise coefficient increases.

This explains why later diffusion states become increasingly dominated by noise.

---

## 16. Common Pitfalls

### 16.1 Markov Does Not Mean Independent

A Markov process can have strong dependencies between consecutive states.

The Markov property only restricts dependence on the more distant past after conditioning on the current state.

### 16.2 Transition Distribution vs Marginal Distribution

The transition distribution

$$
q(x_t\mid x_{t-1})
$$

is not the same as the marginal distribution

$$
q(x_t).
$$

Even if every transition is Gaussian, the marginal distribution need not be Gaussian when the initial data distribution is non-Gaussian.

### 16.3 Conditional Posterior vs Unconditional Reverse Transition

The exact forward posterior

$$
q(x_{t-1}\mid x_t,x_0)
$$

is Gaussian.

The reverse conditional

$$
q(x_{t-1}\mid x_t)
$$

is generally more complicated.

The distinction matters because the original clean sample is not known during generation.

### 16.4 Do Not Confuse Alpha and Alpha-Bar

In DDPM,

$$
\alpha_t=1-\beta_t,
$$

while

$$
\bar\alpha_t=\prod_{s=1}^t\alpha_s.
$$

The one-step transition uses $\alpha_t$.

The direct noising formula uses $\bar\alpha_t$.

### 16.5 Brownian Motion Is Not Differentiable

Brownian motion has continuous sample paths but is almost surely nowhere differentiable.

Therefore, the notation

$$
dW_t
$$

must be understood through stochastic integration rather than ordinary classical differentiation.

This motivates Itô calculus.

---

## 17. Summary and Connection to DDPM

The central concepts are:

| Concept                 | Main idea                                                    |
| ----------------------- | ------------------------------------------------------------ |
| Stochastic process      | A time-indexed family of random variables                    |
| Sample path             | One realization of the process                               |
| Markov property         | The next state depends on the past through the current state |
| Transition kernel       | Conditional probability law between states                   |
| Chapman–Kolmogorov      | Composition of probabilistic transitions                     |
| Stationary distribution | A distribution invariant under a transition kernel           |
| Time reversal           | Reverse conditional distributions follow Bayes' rule         |
| Gaussian Markov chain   | Markov chain with Gaussian transitions                       |
| Brownian motion         | Continuous-time process with Gaussian independent increments |

The most important diffusion equations derived in this module are:

**Forward transition**

$$
\boxed{
q(x_t\mid x_{t-1})
=
\mathcal N(
\sqrt{\alpha_t}x_{t-1},
\beta_tI
).
}
$$

**Forward joint distribution**

$$
\boxed{
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
}
$$

**Direct noising distribution**

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
).
}
$$

**Reparameterized sample**

$$
\boxed{
x_t=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
}
$$

**Exact forward-process posterior**

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N(
\tilde\mu_t(x_t,x_0),
\tilde\beta_tI
).
}
$$

We now have the mathematical tools required to understand the discrete-time foundations of DDPM.

The next generative modeling module will study:

- The learned reverse diffusion process.
- The DDPM variational objective.
- The relationship between Gaussian KL divergence and noise-prediction MSE.
- Neural-network parameterization of the reverse process.
- Sampling algorithms and DDIM.

A later mathematical module will return to Brownian motion, Itô calculus, stochastic differential equations, and the Fokker–Planck equation.

These continuous-time tools are especially important for score-based diffusion models and stochastic differential equation formulations of generative modeling.

---

## Exercises

1. Prove that a first-order Markov chain satisfies

   $$
   p(x_0,\ldots,x_T)
   =
   p(x_0)\prod_{t=1}^T p(x_t\mid x_{t-1}).
   $$

2. For

   $$
   P=
   \begin{pmatrix}
   0.8&0.2\\
   0.3&0.7
   \end{pmatrix},
   $$

   calculate $P^2$ and find the stationary distribution.

3. Derive the mean and covariance recursion for

   $$
   X_t=A_tX_{t-1}+\epsilon_t,
   \qquad
   \epsilon_t\sim\mathcal N(0,Q_t).
   $$

4. Starting from

   $$
   x_t
   =
   \sqrt{\alpha_t}x_{t-1}
   +
   \sqrt{\beta_t}\epsilon_t,
   $$

   derive

   $$
   q(x_t\mid x_0)
   =
   \mathcal N(
   \sqrt{\bar\alpha_t}x_0,
   (1-\bar\alpha_t)I
   ).
   $$

5. Use Gaussian multiplication to derive the conditional variance

   $$
   \tilde\beta_t
   =
   \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}
   \beta_t.
   $$

6. Explain why the exact Gaussian distribution $q(x_{t-1}\mid x_t,x_0)$ cannot directly replace the learned generative transition $p_\theta(x_{t-1}\mid x_t)$ during sampling.

---

## References

1. Norris, J. R. (1997). *Markov Chains*. Cambridge University Press.

2. Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press.

3. Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.

4. Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction with Applications*. Springer.

5. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.

6. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.