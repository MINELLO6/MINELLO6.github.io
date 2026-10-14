---
layout: page
title: Generative Models
permalink: /generative-models/
includelink: false
mathjax: true
---

Generative modeling studies how to learn the underlying
probability distribution of data and generate new samples.

This page organizes my notes on the mathematical
foundations, classical probabilistic models, and
modern deep generative modeling methods.

## Contents

- [00 - Mathematical Foundations](#foundations)
- [01 - Classical Probabilistic Models](#classical)
- [02 - Latent Variable & Likelihood Models](#latent)
- [03 - Energy-Based Models](#energy)
- [04 - Diffusion & Score-Based Models](#diffusion)
- [05 - Flow & Optimal Transport](#flow)
- [06 - Bridge Models](#bridge)
- [07 - Discrete Generative Models](#discrete)
- [08 - Modern Architectures & Acceleration](#modern)

---


## 00 - Mathematical Foundations {#foundations}

**Goal:** Develop the mathematical tools required to understand,
derive, and implement modern generative models.

### M01 - Probability & Information Theory

- Joint, marginal, and conditional distributions
- Conditional expectation and the tower property
- Entropy, differential entropy, and cross-entropy
- KL divergence $D_{\mathrm{KL}}(p\|q)$
- Mutual information $I(X;Y)$
- Change of variables and pushforward distributions

**Applications:** VAE, autoregressive models, normalizing flows.

### M02 - Statistical Inference & Latent Variables

- Maximum likelihood estimation (MLE)
- Maximum a posteriori estimation (MAP)
- Bayesian inference and posterior predictive distributions
- Latent variable models and marginal likelihood
- Expectation-Maximization (EM)

**Applications:** GMM, VAE, Bayesian generative models.

### M03 - Variational Inference

- Jensen's inequality
- Evidence Lower Bound (ELBO)
- KL decomposition and variational objectives
- Mean-field variational inference
- Reparameterization trick

**Core identity:**

$$
\log p_\theta(x)
=
\mathcal{L}_{\mathrm{ELBO}}(x)
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\|p_\theta(z\mid x)
\right)
$$

**Applications:** VAE, DDPM, latent variable models.

### M04 - Gaussian Identities & Matrix Calculus

- Multivariate Gaussian distributions
- Conditional and marginal Gaussian distributions
- Products of Gaussian densities
- KL divergence between Gaussian distributions
- Gradient, Jacobian, Hessian, and trace identities
- Matrix determinant and change-of-variables formula

**Applications:** VAE, DDPM, normalizing flows.

### M05 - Stochastic Processes

- Markov chains and transition kernels
- Brownian motion $W_t$ and quadratic variation
- Itô's lemma and stochastic integration
- Stochastic differential equations (SDEs)
- Langevin dynamics

**Core equation:**

$$
dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t
$$

**Applications:** DDPM, score-based generative models.

### M06 - SDEs & Fokker-Planck Equations

- Forward and reverse-time SDEs
- Fokker-Planck equation
- Score functions $\nabla_x \log p_t(x)$
- Score matching and denoising score matching
- Probability flow ODE
- Connections between SDEs and deterministic flows

**Applications:** Score SDE, NCSN, diffusion models.

### M07 - ODEs, PDEs & Continuous Flows

- Ordinary differential equations (ODEs)
- Continuity equation
- Pushforward probability measures
- Instantaneous change-of-variables formula
- Neural ODEs
- Continuous normalizing flows (CNFs)

**Core equation:**

$$
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot\left(p_t(x)v_t(x)\right)
=0
$$

**Applications:** CNF, flow matching, rectified flow.

### M08 - Optimal Transport & Stochastic Control

- Monge and Kantorovich formulations
- Wasserstein distances $W_p(\mu,\nu)$
- Dynamic optimal transport
- Entropic optimal transport
- Girsanov theorem
- Doob $h$-transform
- Stochastic optimal control

**Applications:** OT-CFM, Schrodinger bridges, diffusion bridges.

### M09 - Numerical Methods

- Euler and Runge-Kutta methods
- Euler-Maruyama method
- Numerical integration of reverse-time SDEs
- Predictor-corrector sampling
- Numerical stability and discretization error
- Adaptive-step ODE solvers

**Applications:** DDPM, DDIM, flow matching, diffusion samplers.

---

### Suggested Learning Path

**Stage 1 - Variational Generative Models**

M01 $\rightarrow$ M02 $\rightarrow$ M03 $\rightarrow$ M04
$\rightarrow$ VAE

**Stage 2 - Diffusion & Score-Based Models**

M04 $\rightarrow$ M05 $\rightarrow$ M06 $\rightarrow$ M09
$\rightarrow$ DDPM / Score SDE

**Stage 3 - Flow & Bridge Models**

M07 $\rightarrow$ M08 $\rightarrow$ M09
$\rightarrow$ Flow Matching / Schrodinger Bridge


## 01 - Classical Probabilistic Models {#classical}

Classical statistical models and probabilistic
generative processes.

- Gaussian Mixture Models (GMM)
- Hidden Markov Models (HMM)
- Bayesian Networks
- Latent Dirichlet Allocation (LDA)
- Expectation-Maximization (EM)

## 02 - Latent Variable & Likelihood Models {#latent}

Deep generative models based on latent variables,
explicit likelihoods, or implicit sampling.

- Variational Autoencoders (VAE)
- Beta-VAE
- Vector Quantized VAE (VQ-VAE)
- Autoregressive Models
- Generative Adversarial Networks (GAN)
- Wasserstein GAN (WGAN)
- Normalizing Flows
- RealNVP & Glow

## 03 - Energy-Based Models {#energy}

Models that represent probability distributions
through energy functions.

- Energy-Based Models (EBM)
- Restricted Boltzmann Machines
- Contrastive Divergence
- Noise Contrastive Estimation
- Langevin Sampling

## 04 - Diffusion & Score-Based Models {#diffusion}

Generative modeling through iterative denoising
and score estimation.

- Denoising Diffusion Probabilistic Models (DDPM)
- Denoising Diffusion Implicit Models (DDIM)
- Denoising Score Matching
- Noise Conditional Score Networks (NCSN)
- Score-Based Generative Modeling through SDEs
- Probability Flow ODE
- Elucidating the Design Space (EDM)

## 05 - Flow & Optimal Transport {#flow}

Continuous-time generative modeling through
transport maps and learned vector fields.

- Normalizing Flows
- Continuous Normalizing Flows (CNF)
- Neural Ordinary Differential Equations
- Optimal Transport
- Flow Matching
- Conditional Flow Matching
- Rectified Flow
- OT-CFM

## 06 - Bridge Models {#bridge}

Stochastic processes connecting endpoints
or probability distributions.

- Brownian Bridges
- Diffusion Bridges
- Schrödinger Bridges
- Diffusion Schrödinger Bridges
- Bridge Matching
- Stochastic Control & Doob h-Transforms

## 07 - Discrete Generative Models {#discrete}

Generative methods for discrete state spaces,
including tokens and categorical variables.

- Discrete Diffusion Models
- D3PM
- Masked Generative Modeling
- MaskGIT
- Discrete Flow Matching
- Diffusion Language Models

## 08 - Modern Architectures & Acceleration {#modern}

Architectural developments and efficient
training or sampling techniques.

- Latent Diffusion Models (LDM)
- Stable Diffusion
- Diffusion Transformers (DiT)
- Consistency Models
- Consistency Trajectory Models
- Progressive Distillation
- Few-Step Generative Models
