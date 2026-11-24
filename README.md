# Single-Cell Gene Expression with Variational Autoencoders

A beginner-friendly PyTorch project comparing **Gaussian VAE, Poisson VAE, and Negative Binomial (NB) VAE** on the PBMC 3k single-cell RNA-seq dataset.

**Research question:** How does the decoder likelihood affect reconstruction, count-distribution modeling, and the learned cell representations?

## Project phases

| Phase | What the code does |
| --- | --- |
| 1. Prepare | Download PBMC 3k, quality control, train/validation/test split, select 2,000 highly variable genes using training cells only. |
| 2. Gaussian VAE | Train a simple VAE on library-normalized log expression; compare reconstruction with PCA. |
| 3. Count VAEs | Train Poisson and NB VAEs on original integer gene counts with per-cell library-size offsets. |
| 4. Evaluate | Held-out negative log likelihood, reconstruction metrics, zero fractions, mean/variance plots, UMAP, and optional multiple seeds and latent dimensions. |
| 5. Report | Save plots, JSON metrics and a Markdown experiment summary for your research write-up. |

## Run on Google Colab

Open a Colab notebook (CPU or GPU), then run:

```bash
!git clone https://github.com/raoankit72005/minor_project.git
%cd minor_project
!pip install -r requirements.txt

!python main.py prepare
!python main.py train --model gaussian
!python main.py train --model poisson
!python main.py train --model nb
!python main.py compare
```

For a larger experiment:

```bash
!python main.py compare --seeds 42 123 7 --latent-dims 10 20 32
```

For a quick self-contained neural-network sanity check, without downloading PBMC data:

```bash
!python main.py smoke-test
```

You can also run each command in a terminal without the Colab `!` prefix.

## What each model receives

- **Gaussian VAE:** encoder uses log(1 + counts normalized to 10,000 *within selected genes*); decoder predicts continuous values and uses unit-variance Gaussian reconstruction loss.
- **Poisson VAE:** encoder uses log(1 + original counts); decoder outputs nonnegative gene means using `softmax` proportions multiplied by observed selected-gene library size.
- **NB VAE:** same setup as Poisson, with an additional learned positive dispersion parameter per gene. NB allows variance to exceed the mean.

All count likelihoods are evaluated against **original raw integer counts** (not normalized/log-transformed values). Their NLL values are comparable to each other on the same test genes; the Gaussian loss is on a different observation scale and should not be directly ranked against count NLL.

## File guide

- `main.py`: all five phases in readable Python functions.
- `requirements.txt`: dependencies.
- `REPORT.md`: explanation of the research question, evaluation plan and limitations.
- `data/`: generated train/validation/test arrays and selected gene names (not committed).
- `outputs/`: checkpoints, plots, metrics and experiment summaries (not committed).

## Important interpretation

The project learns a **compact representation of each cell**, not a cell-type classifier. UMAP groups alone **do not prove** that clusters correspond to biological cell types. PBMC 3k is a single small dataset; random cell splits are not evidence of generalization to new donors or laboratories. The generated profiles are statistical simulations, not experimentally verified cells.

No pretrained models, PyTorch Lightning or scvi-tools trainer are used. This is an educational comparison inspired by probabilistic single-cell models, **not** a reproduction of the full scVI architecture.

## References

- [Scanpy PBMC 3k dataset](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html)
- [Scanpy highly variable gene selection](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html)
- [scVI documentation](https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html)
