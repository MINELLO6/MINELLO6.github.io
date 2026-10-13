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

The mathematical foundations of generative modeling.

- Probability Distributions & Latent Variables
- Maximum Likelihood Estimation
- KL Divergence & Information Theory
- Variational Inference & ELBO
- Markov Chains & Brownian Motion
- Stochastic Differential Equations
- Fokker–Planck & Continuity Equations
- Optimal Transport & Wasserstein Distance
- Numerical ODE & SDE Methods

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
