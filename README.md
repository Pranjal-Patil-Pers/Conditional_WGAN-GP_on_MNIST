\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{xcolor}

\geometry{margin=1in}

\begin{document}

\begin{center}
\Large \textbf{CSC 8851 – Deep Learning (Fall 2025)} \\[6pt]
\Large \textbf{Conditional WGAN-GP on MNIST} \\[4pt]
\end{center}

\vspace{1em}

\section*{Overview}
This assignment implements a \textbf{class-conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)} trained on the \textbf{MNIST dataset (32×32)}. The goal is to generate high-quality, label-conditioned images using a \textbf{projection discriminator} and \textbf{label-conditional generator}, and to explore training stability, latent interpolation, and truncation effects.

\section*{Learning Objectives}
\begin{itemize}
    \item Implement and train a conditional WGAN-GP on MNIST.
    \item Incorporate a projection discriminator and label-conditional generator.
    \item Apply gradient penalty for Lipschitz continuity.
    \item Maintain a Generator EMA (Exponential Moving Average) for cleaner samples.
    \item Visualize loss curves, critic diagnostics, conditional samples, latent interpolations, and truncation sweeps.
\end{itemize}

\section*{Model Architectures}

\subsection*{1. Conditional Generator $G(z, y)$}
\textbf{Inputs:} \\
Noise vector $z \in \mathbb{R}^{64}$ and label embedding $e_g(y) \in \mathbb{R}^{32}$.

\textbf{Architecture:}
\begin{enumerate}
    \item Concatenate $[z, e_g(y)] \rightarrow$ FC: $(64 + 32) \rightarrow 4 \times 4 \times 128$.
    \item Reshape to $(128, 4, 4)$.
    \item Three upsampling blocks with BatchNorm and ReLU:
    \begin{itemize}
        \item Upsample $\times2 \rightarrow$ Conv(128→64) → BN → ReLU.
        \item Upsample $\times2 \rightarrow$ Conv(64→32) → BN → ReLU.
        \item Upsample $\times2 \rightarrow$ Conv(32→16) → BN → ReLU.
    \end{itemize}
    \item Output: Conv(16→1, 3×3) → Tanh.
\end{enumerate}

\textbf{Notes:} Kaiming (He) initialization is used for all Conv and Linear layers. BatchNorm is applied only in the generator.

\subsection*{2. Projection Critic $D(x, y)$}
\textbf{Inputs:} \\
Image $x \in \mathbb{R}^{1 \times 32 \times 32}$ (normalized to [−1,1]) and label $y$.

\textbf{Architecture:}
\begin{enumerate}
    \item Conv(1→32, 4×4, s=2, p=1) → LeakyReLU(0.2)
    \item Conv(32→64, 4×4, s=2, p=1) → LeakyReLU(0.2)
    \item Conv(64→128, 4×4, s=2, p=1) → LeakyReLU(0.2)
    \item Global sum pooling $\Rightarrow f(x) \in \mathbb{R}^{128}$
    \item Projection head:
    \[
    D(x, y) = w^\top f(x) + \langle f(x), e(y) \rangle
    \]
\end{enumerate}

\textbf{Notes:} No sigmoid activation; critic outputs raw Wasserstein scores. No normalization layers (spectral/weight) are used.

\section*{Loss Functions}

\subsection*{Critic Loss (with Gradient Penalty)}
\[
L_C = - \mathbb{E}[D(x,y)] + \mathbb{E}[D(G(z,y),y)] + \lambda_{gp} \, \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x}, y)\|_2 - 1)^2]
\]
where
\[
\hat{x} = \epsilon x + (1 - \epsilon)G(z,y), \quad \epsilon \sim \mathcal{U}(0,1)
\]
and $\lambda_{gp} = 10$.

\subsection*{Generator Loss}
\[
L_G = - \mathbb{E}[D(G(z,y),y)]
\]

\section*{Generator EMA (Exponential Moving Average)}
After every generator update:
\[
\theta_{ema} \leftarrow \tau \theta_{ema} + (1 - \tau)\theta, \quad \tau = 0.999
\]
EMA improves sampling stability and image quality.

\section*{Training Details}

\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimizer & Adam \\
Learning Rate & $2 \times 10^{-4}$ \\
Betas & (0.0, 0.9) \\
$\lambda_{gp}$ & 10 \\
$n_{critic}$ & 2--3 \\
Epochs & 10--20 \\
Batch Size & 128 \\
Device & GPU (recommended) \\
\bottomrule
\end{tabular}
\end{center}

\section*{Plots and Visualizations}
During training, the following are plotted:
\begin{itemize}
    \item Losses: $L_C$, $L_G$
    \item Critic metrics: $\mathbb{E}[D(x,y)]$, $\mathbb{E}[D(G(z,y),y)]$, Gradient Penalty (GP)
\end{itemize}

At the end of training:
\begin{enumerate}
    \item \textbf{Conditional Samples Grid:} 10×N (rows = digits 0–9)
    \item \textbf{Latent Interpolations:} Smooth transitions between two latent vectors per class
    \item \textbf{Truncation Sweep:} Scaling $z \leftarrow \psi z$ with $\psi \in \{3.0, 2.5, \ldots, 0.1\}$ 
\end{enumerate}

\section*{Results Summary}
\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{Visualization} & \textbf{Description} \\
\midrule
Conditional Grid & High-quality digit samples per class using EMA generator. \\
Interpolation Panel & Smooth morphing between latent endpoints, class-consistent. \\
Truncation Sweep & Demonstrates diversity (high $\psi$) vs. fidelity (low $\psi$). \\
\bottomrule
\end{tabular}
\end{center}

\section*{How to Run}
\begin{enumerate}
    \item Install dependencies:
    \begin{verbatim}
    pip install torch torchvision matplotlib tqdm
    \end{verbatim}
    \item Open and run the notebook:
    \begin{verbatim}
    jupyter notebook CSC8851_F2025_HW2_PranjalPatil.ipynb
    \end{verbatim}
    \item The notebook will train $G$ and $D$, save plots, and generate final samples.
\end{enumerate}

\section*{Notes}
\begin{itemize}
    \item Code runs top-to-bottom without manual intervention.
    \item All plots and figures are auto-generated.
    \item No pre-trained models used.
    \item Any external sources are cited in the notebook.
\end{itemize}

\end{document}
