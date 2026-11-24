# Research report outline

## Objective
Study how Gaussian, Poisson and Negative Binomial decoder likelihoods influence reconstructed expression and latent representations in PBMC 3k.

## Biology
A row represents a cell, a column represents a gene, and each element is an observed RNA-derived count. The model **does not predict cell-type names**. It learns compressed numerical representations of cell expression.

## Likelihoods and ELBO
All three use a Gaussian latent prior and diagonal Gaussian approximate posterior.

ELBO = expected log likelihood of observation - KL(q(z|x) || N(0,I)).

- Gaussian: normalized/log counts, fixed variance=1, squared-error negative log likelihood (up to a constant).
- Poisson: raw counts, mean = selected-gene library size times decoder softmax proportions; variance equals mean.
- Negative Binomial: same mean, learnable gene-specific dispersion; variance = mean + mean²/dispersion.

Poisson and NB count NLL values are comparable on the same held-out gene set. **Do not directly compare their raw NLL to Gaussian reconstruction values** because the observations differ.

## Evaluation
Split by cells; select high-variance genes on training cells only; use the same splits for all models.

1. Test negative ELBO, reconstruction term and KL.
2. Compare Poisson vs NB count likelihoods and predicted zero probabilities.
3. Compare gene mean behavior across observed and predicted data.
4. Explore PCA projections of posterior means, not only training loss.
5. Repeat with several seeds and latent dimensions for variability.

## Key caveats
- Posterior means and reconstructed count means are expectations, not observed gene counts; exact count predictions are not expected.
- Sampled counts are noisy, so distributional statistics matter.
- A UMAP/PCA cloud does not establish cell-type separation without biological reference labels.
- Random-cell split performance on one sample does not imply transfer to new donors, protocols, or batches.
- Using observed library sizes in the decoder models a *conditional* distribution; it does not produce unconditional full-library samples.
- Gaussian input uses selected-gene library normalization; count models use raw selected-gene counts and selected-gene size.
- Count NLL evaluated with posterior samples is a reconstruction component of an ELBO, not exact marginal likelihood.
- This small educational project does not claim novel research findings or implement the full scVI model.

## Next extensions
Use meaningful externally annotated cell types for a held-out linear probe, add gene-wise variance calibration, test generated samples against observed distributions, then compare flow-enhanced approximate posteriors if a credible baseline is established.
