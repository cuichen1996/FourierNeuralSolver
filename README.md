
## Fourier Neural Solver (FNS)

This repository provides the official implementation of Fourier Neural Solver (FNS) — a neural network-based hybrid solver for large sparse linear systems.

FNS integrates the smoothing effect of simple iterative methods (e.g., damped Jacobi) with the spectral learning capabilities of neural networks. The neural network architecture is designed from an eigenvalue analysis perspective:
- A meta subnet learns the eigenvalues corresponding to error components that are difficult to eliminate.
- Eigenvectors are approximated using Fourier modes, with a transition matrix predicted by another meta subnet.

As a result, FNS can achieve grid-independent convergence rates and exhibits robustness to variations in PDE parameters.

## References
- Chen Cui, Kai Jiang, Yun Liu, Shi Shu. 2025, A Hybrid Iterative Neural Solver Based on Spectral Analysis for Parametric PDEs. *Journal of Computational Physics*, 114165. https://doi.org/10.1016/j.jcp.2025.114165
- Chen Cui, Kai Jiang, Yun Liu, Shi Shu, 2022. Fourier Neural Solver for Large Sparse Linear Algebraic Systems. *Mathematics*, 10(21), 4014. https://doi.org/10.3390/math10214014

