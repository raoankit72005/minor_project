import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("data")
OUT = Path("outputs")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense(x):
    return (x.toarray() if sparse.issparse(x) else np.asarray(x)).astype("float32")


def prepare():
    DATA.mkdir(exist_ok=True)
    adata = sc.datasets.pbmc3k()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["n_genes_by_counts"] < 2500)
        & (adata.obs["pct_counts_mt"] < 5)
    ].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    rng = np.random.default_rng(42)
    indices = rng.permutation(adata.n_obs)
    n_train = int(len(indices) * 0.7)
    n_val = int(len(indices) * 0.15)
    train = adata[indices[:n_train]].copy()
    val = adata[indices[n_train:n_train + n_val]].copy()
    test = adata[indices[n_train + n_val:]].copy()

    sc.pp.highly_variable_genes(train, n_top_genes=2000, flavor="seurat_v3")
    genes = train.var_names[train.var["highly_variable"]]
    np.save(DATA / "genes.npy", genes.to_numpy())

    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = dense(split[:, genes].X)
        assert np.all(counts >= 0) and np.allclose(counts, np.round(counts))
        np.save(DATA / f"{name}.npy", counts)
        np.save(DATA / f"{name}_total_counts.npy", split.obs["total_counts"].to_numpy())
        print(name, counts.shape)
    print("Saved data to", DATA)


def load_data():
    if not (DATA / "train.npy").exists():
        raise FileNotFoundError("Run: python main.py prepare")
    return {name: np.load(DATA / f"{name}.npy") for name in ["train", "val", "test"]}


def inputs(x, kind):
    if kind == "gaussian":
        totals = np.maximum(x.sum(axis=1, keepdims=True), 1)
        return np.log1p(x / totals * 10000).astype("float32")
    return np.log1p(x).astype("float32")


class VAE(nn.Module):
    def __init__(self, n_genes, latent_dim, kind):
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_genes)
        )
        if kind == "nb":
            self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-12, 12)

    def forward(self, x, library):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        output = self.decoder(z)
        if self.kind != "gaussian":
            output = torch.softmax(output, dim=1) * library[:, None]
            output = output.clamp(min=1e-8)
        return output, mu, logvar


def poisson_nll(x, mean):
    return mean - x * torch.log(mean) + torch.lgamma(x + 1)


def nb_nll(x, mean, theta):
    return (
        torch.lgamma(theta) + torch.lgamma(x + 1)
        - torch.lgamma(x + theta)
        + theta * (torch.log(theta + mean) - torch.log(theta))
        + x * (torch.log(theta + mean) - torch.log(mean))
    )


def losses(model, x, output, mu, logvar):
    if model.kind == "gaussian":
        reconstruction = 0.5 * (output - x).square().sum(dim=1).mean()
    elif model.kind == "poisson":
        reconstruction = poisson_nll(x, output).sum(dim=1).mean()
    else:
        theta = torch.nn.functional.softplus(model.log_theta).clamp(min=1e-4)
        reconstruction = nb_nll(x, output, theta).sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return reconstruction + kl, reconstruction, kl


def loader(x, kind, batch_size, shuffle=False):
    encoder_x = torch.from_numpy(inputs(x, kind))
    target = torch.from_numpy(inputs(x, kind) if kind == "gaussian" else x)
    library = torch.from_numpy(x.sum(axis=1).astype("float32").clip(min=1))
    return DataLoader(TensorDataset(encoder_x, target, library),
                      batch_size=batch_size, shuffle=shuffle)


def evaluate(model, x, batch_size=128, samples=3):
    model.eval()
    total = np.zeros(3)
    device = next(model.parameters()).device
    with torch.no_grad():
        for enc, target, library in loader(x, model.kind, batch_size):
            enc, target, library = enc.to(device), target.to(device), library.to(device)
            values = np.zeros(3)
            for _ in range(samples):
                output, mu, logvar = model(enc, library)
                result = losses(model, target, output, mu, logvar)
                values += np.array([v.item() for v in result]) / samples
            total += values * len(enc)
    return dict(zip(["loss", "reconstruction", "kl"], (total / len(x)).tolist()))


def fit(kind, latent_dim=20, seed=42, epochs=100, batch_size=128):
    set_seed(seed)
    data = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(data["train"].shape[1], latent_dim, kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    run = OUT / f"{kind}_z{latent_dim}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)

    best, patience = float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train()
        sums = np.zeros(3)
        for enc, target, library in loader(data["train"], kind, batch_size, True):
            enc, target, library = enc.to(device), target.to(device), library.to(device)
            output, mu, logvar = model(enc, library)
            total, recon, kl = losses(model, target, output, mu, logvar)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            sums += np.array([total.item(), recon.item(), kl.item()]) * len(enc)
        validation = evaluate(model, data["val"], batch_size)
        train_loss = (sums / len(data["train"])).tolist()
        history.append([epoch + 1, *train_loss, validation["loss"]])
        print(f"{kind} epoch {epoch+1}: train={train_loss[0]:.2f} val={validation['loss']:.2f}")
        if validation["loss"] < best:
            best = validation["loss"]
            patience = 0
            torch.save(model.state_dict(), run / "model.pt")
        else:
            patience += 1
        if patience >= 15:
            break

    model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
    metrics = evaluate(model, data["test"], batch_size, samples=10)
    metrics.update(model=kind, latent_dim=latent_dim, seed=seed,
                   n_test=len(data["test"]), epochs=len(history))
    with open(run / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savetxt(run / "history.csv", history, delimiter=",",
               header="epoch,train_loss,train_reconstruction,train_kl,val_loss", comments="")
    fig, ax = plt.subplots()
    ax.plot([h[0] for h in history], [h[1] for h in history], label="train")
    ax.plot([h[0] for h in history], [h[4] for h in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Negative ELBO", title=kind.upper())
    ax.legend()
    fig.savefig(run / "loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with torch.no_grad():
        x = torch.from_numpy(inputs(data["test"], kind)).to(device)
        z = model.encode(x)[0].cpu().numpy()
    np.save(run / "test_latent.npy", z)
    if kind == "gaussian":
        x_train = inputs(data["train"], kind)
        x_test = inputs(data["test"], kind)
        pca = PCA(n_components=min(latent_dim, len(x_train) - 1))
        pca.fit(x_train)
        predicted = pca.inverse_transform(pca.transform(x_test))
        metrics["pca_test_mse"] = float(np.mean((x_test - predicted) ** 2))
        with open(run / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    print("Test:", metrics)
    return metrics


def plot_embedding(z, filename, title):
    from sklearn.decomposition import PCA
    if len(z) < 3:
        return
    coords = PCA(n_components=2).fit_transform(z)
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap(z, filename, title):
    try:
        import umap
    except ImportError:
        return
    coords = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(z)
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnostics(kind, latent_dim, seed):
    data = load_data()
    run = OUT / f"{kind}_z{latent_dim}_seed{seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(data["test"].shape[1], latent_dim, kind).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x = data["test"]
    with torch.no_grad():
        encoder_input = torch.from_numpy(inputs(x, kind)).to(device)
        library = torch.from_numpy(x.sum(axis=1).astype("float32").clip(min=1)).to(device)
        z = model.encode(encoder_input)[0]
        prediction = model.decoder(z)
        if kind != "gaussian":
            prediction = torch.softmax(prediction, dim=1) * library[:, None]
        prediction = prediction.cpu().numpy()
    plot_embedding(z.cpu().numpy(), run / "latent_pca.png", kind + " latent PCA")
    plot_umap(z.cpu().numpy(), run / "latent_umap.png", kind + " latent UMAP")
    if kind == "gaussian":
        return
    selected = np.argsort(x.mean(axis=0))[-30:]
    fig, ax = plt.subplots()
    ax.scatter(x[:, selected].mean(axis=0), prediction[:, selected].mean(axis=0))
    ax.set(xlabel="Observed gene mean", ylabel="Predicted gene mean",
           title=kind + " gene means (30 abundant genes)")
    fig.savefig(run / "gene_means.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    observed_zeros = float((x == 0).mean())
    if kind == "poisson":
        predicted_zeros = float(np.exp(-prediction).mean())
    else:
        theta = torch.nn.functional.softplus(model.log_theta).detach().cpu().numpy()
        predicted_zeros = float(np.mean((theta / (theta + prediction)) ** theta))
    with open(run / "diagnostics.json", "w") as f:
        json.dump({"observed_zero_fraction": observed_zeros,
                   "predicted_zero_probability": predicted_zeros}, f, indent=2)


def compare(seeds, latent_dims, epochs):
    rows = []
    for latent_dim in latent_dims:
        for seed in seeds:
            for kind in ["gaussian", "poisson", "nb"]:
                rows.append(fit(kind, latent_dim, seed, epochs))
                diagnostics(kind, latent_dim, seed)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    lines = [
        "# Experiment results", "",
        "Models use the same held-out cells and genes.",
        "Gaussian reconstruction values are on normalized log-expression,",
        "while Poisson and NB values are raw count negative log likelihoods.",
        "Do not compare those absolute numbers across scales.", "",
        "| Model | Latent | Seed | Test reconstruction | KL |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['latent_dim']} | {r['seed']} | {r['reconstruction']:.2f} | {r['kl']:.2f} |")
    (OUT / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")
    print("Saved comparison to outputs/comparison.json and outputs/EXPERIMENTS.md")


def smoke_test():
    set_seed(42)
    x = torch.poisson(torch.full((8, 16), 2.0))
    for kind in ["gaussian", "poisson", "nb"]:
        model = VAE(16, 4, kind)
        enc = torch.log1p(x) if kind != "gaussian" else torch.log1p(x / x.sum(1, keepdim=True) * 10000)
        target = enc if kind == "gaussian" else x
        output, mu, logvar = model(enc, x.sum(1))
        loss, _, _ = losses(model, target, output, mu, logvar)
        assert torch.isfinite(loss), kind
        loss.backward()
        assert model.mu.weight.grad is not None
        print(kind, "OK", round(loss.item(), 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "compare", "smoke-test"])
    parser.add_argument("--model", choices=["gaussian", "poisson", "nb"], default="gaussian")
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[20])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        fit(args.model, args.latent_dim, args.seed, args.epochs)
        diagnostics(args.model, args.latent_dim, args.seed)
    elif args.command == "compare":
        compare(args.seeds, args.latent_dims, args.epochs)
    else:
        smoke_test()


if __name__ == "__main__":
    main()
