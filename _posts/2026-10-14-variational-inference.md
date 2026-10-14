---
layout: post
title: "M03 - Variational Inference: ELBO, Mean-Field and Reparameterization"
date: 2026-10-14
mathjax: true
comments: true
---

**Prerequisite:** [M02 - Statistical Inference and Latent Variable Models]({{ '/2026/10/13/statistical-inference/' | relative_url }}).  
**Aim:** understand variational inference as an optimization approach to approximate Bayesian posterior inference, and derive the mathematical tools behind VAEs.

## 1. Why Variational Inference?

Suppose a latent-variable model has joint density

$$
p_\theta(x,z)=p_\theta(x\mid z)p(z).
$$

Given an observation $x$, Bayesian inference asks for the posterior

$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)},
\qquad
p_\theta(x)=\int p_\theta(x,z)\,dz.
$$

For a complex decoder $p_\theta(x\mid z)$, this integral is often intractable. We might be able to evaluate the joint density at a given $(x,z)$ while being unable to evaluate its normalizing constant $p_\theta(x)$.

**Variational inference (VI)** replaces exact posterior computation with an optimization problem: choose an approximating distribution $q(z)$ from a tractable family $\mathcal Q$.

$$
\boxed{
q^*=\operatorname*{arg\,min}_{q\in\mathcal Q}
D_{\mathrm{KL}}\bigl(q(z)\|p_\theta(z\mid x)\bigr).
}
$$

This is called **reverse KL** because the approximate distribution comes first. Crucially, VI can optimize an equivalent objective without computing the unknown evidence $p_\theta(x)$.

## 2. Deriving the Evidence Lower Bound

Introduce a density $q(z)$ with suitable support and well-defined expectations. Then

$$
\begin{aligned}
\log p_\theta(x)
&=\log\int q(z)\frac{p_\theta(x,z)}{q(z)}\,dz\\
&=\log\mathbb E_q\left[\frac{p_\theta(x,Z)}{q(Z)}\right]\\
&\geq\mathbb E_q[\log p_\theta(x,Z)-\log q(Z)],
\end{aligned}
$$

where the inequality follows from Jensen's inequality, since $\log$ is concave.

Define the **evidence lower bound (ELBO)**:

$$
\boxed{
\mathcal L(q,\theta;x)
=\mathbb E_{q(z)}[\log p_\theta(x,z)-\log q(z)].
}
$$

Equivalently, $\mathcal L=\mathbb E_q[\log p_\theta(x,Z)]+H(q)$, where $H(q)=-\mathbb E_q[\log q(Z)]$.

Using Bayes' rule, we obtain an *exact* decomposition:

$$
\boxed{
\log p_\theta(x)
=\mathcal L(q,\theta;x)
+D_{\mathrm{KL}}\bigl(q(z)\|p_\theta(z\mid x)\bigr).
}
$$

For fixed $\theta$, the log evidence is constant as $q$ changes. Therefore

$$
\boxed{
\operatorname*{arg\,max}_{q\in\mathcal Q}\mathcal L(q,\theta;x)
=
\operatorname*{arg\,min}_{q\in\mathcal Q}
D_{\mathrm{KL}}(q\|p_\theta(z\mid x)).
}
$$

The lower bound is **tight** if and only if $q(z)=p_\theta(z\mid x)$ almost everywhere. In an unrestricted family, exact Bayesian posterior inference is the optimum of the variational objective.

### 2.1 Reconstruction and regularization

Substitute $p_\theta(x,z)=p_\theta(x\mid z)p(z)$:

$$
\begin{aligned}
\mathcal L(q,\theta;x)
&=\mathbb E_q[\log p_\theta(x\mid Z)]\\
&\quad+\mathbb E_q[\log p(Z)-\log q(Z)].
\end{aligned}
$$

Thus

$$
\boxed{
\mathcal L(q,\theta;x)
=
\underbrace{\mathbb E_q[\log p_\theta(x\mid Z)]}_{\text{expected reconstruction log-likelihood}}
-
\underbrace{D_{\mathrm{KL}}(q(z)\|p(z))}_{\text{prior regularization}}.
}
$$

These two terms have distinct roles. The first rewards latent codes that allow the decoder to explain $x$. The second penalizes a posterior approximation that differs from the prior.

**Do not confuse the two KL divergences:**

- $D_{\mathrm{KL}}(q(z)\|p_\theta(z\mid x))$ is the **inference gap** between the ELBO and $\log p_\theta(x)$.
- $D_{\mathrm{KL}}(q(z)\|p(z))$ is the **prior regularizer** appearing *inside* the ELBO.

They are generally not the same quantity.

## 3. Why the Direction of KL Matters

The reverse-KL objective is

$$
D_{\mathrm{KL}}(q\|p)=\mathbb E_q\left[\log\frac{q(Z)}{p(Z)}\right].
$$

It penalizes placing mass where the target $p$ has very little mass. In a restricted unimodal approximating family and a multimodal target, this may encourage $q$ to concentrate on one mode.

In comparison, forward KL is

$$
D_{\mathrm{KL}}(p\|q)=\mathbb E_p\left[\log\frac{p(Z)}{q(Z)}\right],
$$

and heavily penalizes failing to cover regions where $p$ assigns mass. This can encourage mode-covering behavior.

These tendencies depend on the target and variational family; neither direction is always mode-seeking or mode-covering. If $q=p$ is representable, both divergences attain zero there.

**Why use reverse KL in standard VI?** The ELBO rearranges directly into a reverse KL plus the constant $\log p_\theta(x)$. We can evaluate or estimate the ELBO from $q$ and the unnormalized joint density without calculating the posterior normalizer.

## 4. Mean-Field Variational Inference

For a high-dimensional latent variable $z=(z_1,\ldots,z_m)$, choose the factorized family

$$
\boxed{
q(z)=\prod_{j=1}^m q_j(z_j).
}
$$

This is the **mean-field approximation**: the factors are independent under $q$, even if the true posterior contains dependencies.

We can optimize one factor $q_j$ while keeping all the others fixed. Write $z_{-j}$ for all latent variables except $z_j$.

Ignoring terms independent of $q_j$, the ELBO is

$$
\mathcal L(q)
=
\int q_j(z_j)
\mathbb E_{q_{-j}}[\log p_\theta(x,z)]\,dz_j
-\int q_j(z_j)\log q_j(z_j)\,dz_j+C.
$$

Introduce a Lagrange multiplier to enforce $\int q_j(z_j)\,dz_j=1$. Setting the functional derivative to zero yields

$$
\log q_j^*(z_j)
=
\mathbb E_{q_{-j}}[\log p_\theta(x,z)]+C_j.
$$

Exponentiating and normalizing gives the **coordinate-ascent variational inference (CAVI) update**:

$$
\boxed{
q_j^*(z_j)\propto
\exp\left(
\mathbb E_{q_{-j}}[\log p_\theta(x,z)]
\right).
}
$$

The expectation is over the *other* approximate latent factors; $z_j$ is left free. We repeatedly update each factor until the ELBO or factor parameters stabilize.

### 4.1 Example: approximating a correlated Gaussian

Suppose the target posterior is

$$
p(z\mid x)=\mathcal N\left(
\begin{bmatrix}0\\0\end{bmatrix},
\begin{bmatrix}1&\rho\\ \rho&1\end{bmatrix}
\right), \quad |\rho|<1.
$$

Restrict the approximation to independent Gaussians $q(z)=q_1(z_1)q_2(z_2)$. By the CAVI update, the optimum is

$$
\boxed{
q^*(z_1,z_2)
=
\mathcal N(z_1;0,1-\rho^2)
\mathcal N(z_2;0,1-\rho^2).
}
$$

To see why, note that the target log-density contains

$$
\log p(z\mid x)=
-\frac{z_1^2-2\rho z_1z_2+z_2^2}{2(1-\rho^2)}+C.
$$

Taking expectation over $q_2$ gives a Gaussian log-factor for $z_1$; by symmetry the coordinate-optimal means are zero and both variances are $1-\rho^2$.

For example, when $\rho=0.9$, each variational marginal has variance $0.19$, whereas each true marginal has variance $1$. Mean-field reverse-KL approximation can be substantially **under-dispersed** because it cannot represent the correlation.

## 5. From Local Variational Factors to Amortized Inference

In conventional VI, we may optimize a separate $q_i(z)$ for every data point $x_i$:

$$
\max_{\theta,\{q_i\}}\sum_{i=1}^n
\mathcal L(q_i,\theta;x_i).
$$

For large datasets, repeatedly optimizing local distributions is expensive. **Amortized variational inference** replaces them with a shared inference network:

$$
\boxed{
q_\phi(z\mid x).
}
$$

The network takes $x$ as input and predicts the parameters of its approximate posterior. The same parameters $\phi$ are shared across observations.

For example, a Gaussian encoder produces

$$
q_\phi(z\mid x)=
\mathcal N\left(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right).
$$

This permits efficient inference for a new $x$ with one forward pass, but restricts the local posterior estimates to those representable by the encoder. The resulting additional discrepancy relative to individually optimized factors is often called the **amortization gap**.

## 6. The Reparameterization Trick

To train an encoder, we need to differentiate an expectation involving the encoder's parameters:

$$
\nabla_\phi\,
\mathbb E_{q_\phi(z\mid x)}[f(z)].
$$

The challenge is that the *distribution used to draw $z$* depends on $\phi$. Simply sampling $z$ and treating it as a constant during backpropagation would omit the dependence of the expectation on $\phi$.

For a diagonal Gaussian, write

$$
z\sim \mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).
$$

Draw parameter-independent noise instead:

$$
\epsilon\sim\mathcal N(0,I),
\qquad
\boxed{
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
}
$$

Now the stochasticity lives in $\epsilon$, whose distribution does **not** depend on $\phi$. Consequently,

$$
\boxed{
\nabla_\phi\mathbb E_{q_\phi(z\mid x)}[f(z)]
=
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[\nabla_\phi f\bigl(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\bigr)\right],
}
$$

assuming $f$ has no separate explicit dependence on $\phi$ and differentiation can be interchanged with expectation. If $f$ explicitly depends on $\phi$, include its direct derivative as well.

In practice, estimate this expectation with one or several noise draws and backpropagate through the deterministic transformation.

**Why not sample $z$ directly and differentiate?** Sampling from a parameterized distribution is not, by itself, a differentiable path from $z$ back to the distribution parameters. Reparameterization makes that dependence explicit.

### 6.1 Gaussian KL in closed form

A VAE often takes $p(z)=\mathcal N(0,I)$. For the diagonal Gaussian encoder above,

$$
\boxed{
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\|\mathcal N(0,I)\bigr)
=
\frac12\sum_{j=1}^d
\left(\mu_{\phi,j}(x)^2+\sigma_{\phi,j}(x)^2
-1-\log\sigma_{\phi,j}^2(x)\right).
}
$$

The KL can be calculated analytically, while the reconstruction expectation is commonly approximated by Monte Carlo samples obtained via reparameterization.

## 7. The VAE Objective

A **variational autoencoder (VAE)** is a latent-variable generative model with

- a **prior** $p(z)$, often $\mathcal N(0,I)$;
- a **decoder / likelihood model** $p_\theta(x\mid z)$;
- an **encoder / inference model** $q_\phi(z\mid x)$.

Its per-observation training objective is

$$
\boxed{
\mathcal L_{\mathrm{VAE}}(\theta,\phi;x)
=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z)).
}
$$

The encoder infers plausible latent codes from observations; the decoder specifies how observed data are generated from those codes. To generate a new sample after training, draw $z\sim p(z)$ and then $x\sim p_\theta(x\mid z)$.

For a Bernoulli decoder, maximizing expected log-likelihood is related to minimizing binary cross-entropy. For a Gaussian decoder with **fixed isotropic variance**, it is equivalent up to constants and scaling to minimizing squared reconstruction error. Thus the commonly used "reconstruction loss" depends on the probabilistic observation model.

### 7.1 Training step in pseudocode

```python
# x: a minibatch; encoder and decoder are differentiable networks
mu, logvar = encoder(x)
std = (0.5 * logvar).exp()
eps = torch.randn_like(std)
z = mu + std * eps                    # reparameterization

reconstruction_log_prob = decoder_log_prob(x, z)
kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=-1)

elbo = reconstruction_log_prob - kl   # one value per observation
loss = -elbo.mean()
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Here `decoder_log_prob` must return a properly specified per-sample log-likelihood summed over the observation dimensions, not an arbitrary unnormalized pixel error.

### 7.2 VI, variational EM and VAE: the distinction

| Method | Posterior approximation | Typical optimization |
|---|---|---|
| Exact EM | Exact posterior at current parameters | Exact E-step, then M-step |
| Variational EM | Optimized $q\in\mathcal Q$ | Alternating optimizations over $q$ and $\theta$ |
| VAE | Shared neural encoder $q_\phi(z\mid x)$ | Usually minibatch stochastic gradient ascent on an ELBO |

A VAE uses the **variational principle** of variational EM but is not required to perform exact alternating maximizations. With stochastic minibatch updates, neither the ELBO nor the observed-data likelihood must increase at every single step.

## 8. Practical Limitations and Connections

**Variational approximation gap.** A restricted family such as diagonal Gaussians might not represent the true posterior well.

**Amortization gap.** Even inside the chosen family, an encoder may fail to output the individually optimal variational distribution.

**Posterior collapse.** A sufficiently powerful decoder may largely ignore $z$, with $q_\phi(z\mid x)$ approaching the prior. A small KL term is not, by itself, evidence of a useful latent representation.

**Importance-weighted bounds.** Methods such as IWAE use multiple importance samples to construct a tighter lower bound under appropriate conditions, though a tighter training bound does not automatically solve every inference or representation problem.

**Diffusion models.** DDPM-style objectives can also be derived from variational bounds on latent Markov chains, although their latent structure and training objectives differ from the elementary VAE presented here.

## 9. What to Remember Before M04 and VAE

1. VI replaces exact posterior inference by optimization in a tractable variational family.
2. The ELBO differs from the log evidence by **reverse KL to the true posterior**.
3. Mean-field VI gives coordinate updates of the form $q_j^*\propto\exp(\mathbb E_{q_{-j}}[\log p(x,z)])$.
4. Amortization uses one inference network instead of optimizing a separate variational distribution for every observation.
5. The Gaussian reparameterization $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$ enables low-variance pathwise gradient estimates.
6. VAE training maximizes a reconstruction expectation minus a KL penalty to the prior.

**Self-check:** Derive both the ELBO decomposition and the CAVI update. Explain why the KL to the prior in the VAE loss is different from the KL to the posterior measuring the variational gap.

## References

- Jordan et al., *An Introduction to Variational Methods for Graphical Models* (1999).
- Blei, Kucukelbir, and McAuliffe, *Variational Inference: A Review for Statisticians* (2017).
- Kingma and Welling, *Auto-Encoding Variational Bayes* (2013/2014).
- Rezende, Mohamed, and Wierstra, *Stochastic Backpropagation and Approximate Inference in Deep Generative Models* (2014).
- Burda, Grosse, and Salakhutdinov, *Importance Weighted Autoencoders* (2015).
