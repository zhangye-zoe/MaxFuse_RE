#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, io as spio


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def pick_count_layer(adata: ad.AnnData) -> np.ndarray:
    for key in ["count", "counts", "data"]:
        if key in adata.layers:
            return to_dense(adata.layers[key])
    return to_dense(adata.X)


def pick_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]


def read_fixed_cell_ids(path):
    if path is None:
        return None
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(x)
    return ids


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_azimuth_label_df(csv_path, label_col="predicted.celltype.l2"):
    """
    Read Azimuth annotation csv and return a label_df indexed by cell_id.
    Keeps all available predicted columns and creates a unified 'cell_type' column.
    """
    if csv_path is None:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Azimuth annotation file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "cell_id" not in df.columns:
        raise KeyError("Azimuth csv must contain column 'cell_id'.")

    keep_cols = ["cell_id"]
    pred_cols = [
        "predicted.celltype.l1.score",
        "predicted.celltype.l1",
        "predicted.celltype.l2.score",
        "predicted.celltype.l2",
        "predicted.celltype.l3.score",
        "predicted.celltype.l3",
    ]
    keep_cols += [c for c in pred_cols if c in df.columns]

    df = df[keep_cols].copy()
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.drop_duplicates(subset="cell_id").set_index("cell_id")

    if label_col not in df.columns:
        raise KeyError(
            f"Requested azimuth label column {label_col!r} not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    df["cell_type"] = df[label_col].astype(str)
    return df


def build_label_df_from_obs(obs_df):
    """
    Build label df from existing obs columns if available.
    """
    obs_df = obs_df.copy()
    if "cell_id" not in obs_df.columns:
        return None

    label_candidates = [
        "celltype", "cell_type", "annotation", "annot", "labels", "leiden",
        "future_label", "batch"
    ]
    label_cols = pick_existing_columns(obs_df, label_candidates)
    if len(label_cols) == 0:
        return None

    label_df = obs_df[["cell_id"] + label_cols].copy()
    label_df["cell_id"] = label_df["cell_id"].astype(str)
    label_df = label_df.drop_duplicates(subset="cell_id").set_index("cell_id")
    return label_df


def merge_label_dfs(primary_df, secondary_df):
    """
    Merge two label dfs by index.
    Preference: keep primary columns when duplicated, and fill missing cell_type from secondary.
    """
    if primary_df is None and secondary_df is None:
        return None
    if primary_df is None:
        return secondary_df.copy()
    if secondary_df is None:
        return primary_df.copy()

    out = primary_df.copy()

    for col in secondary_df.columns:
        if col not in out.columns:
            out[col] = secondary_df[col]

    out = out.combine_first(secondary_df)

    if "cell_type" not in out.columns and "cell_type" in secondary_df.columns:
        out["cell_type"] = secondary_df["cell_type"]
    elif "cell_type" in out.columns and "cell_type" in secondary_df.columns:
        out["cell_type"] = out["cell_type"].where(
            out["cell_type"].notna(), secondary_df["cell_type"]
        )

    return out


def align_to_target_ids(expr_cells, meta_cells, label_df, target_ids, strict=True):
    expr_cells = expr_cells.copy()
    meta_cells = meta_cells.copy()

    expr_cells.obs["cell_id"] = expr_cells.obs["cell_id"].astype(str)
    meta_cells.obs["cell_id"] = meta_cells.obs["cell_id"].astype(str)

    expr_cells = expr_cells[~expr_cells.obs["cell_id"].duplicated()].copy()
    meta_cells = meta_cells[~meta_cells.obs["cell_id"].duplicated()].copy()

    expr_cells.obs = expr_cells.obs.set_index("cell_id", drop=False)
    meta_obs = meta_cells.obs.copy().set_index("cell_id", drop=False)

    if label_df is not None:
        label_df = label_df.copy()
        label_df.index = label_df.index.astype(str)
        meta_obs = meta_obs.join(label_df, how="left", rsuffix="_label")

    expr_ids = set(expr_cells.obs.index.astype(str))
    meta_ids = set(meta_obs.index.astype(str))
    label_ids = set(label_df.index.astype(str)) if label_df is not None else set(target_ids)

    missing_expr = [x for x in target_ids if x not in expr_ids]
    missing_meta = [x for x in target_ids if x not in meta_ids]
    missing_label = [x for x in target_ids if x not in label_ids]

    if strict and (missing_expr or missing_meta or missing_label):
        msg = [
            f"Target cells requested: {len(target_ids)}",
            f"Present in expr_cells: {len(expr_ids)}",
            f"Present in meta_cells: {len(meta_ids)}",
            f"Present in label_df: {len(label_ids)}",
            f"Missing in expr_cells: {len(missing_expr)}",
            f"Missing in meta_cells: {len(missing_meta)}",
            f"Missing in label_df: {len(missing_label)}",
        ]
        if missing_expr:
            msg.append(f"First few missing expr ids: {missing_expr[:10]}")
        if missing_meta:
            msg.append(f"First few missing meta ids: {missing_meta[:10]}")
        if missing_label:
            msg.append(f"First few missing label ids: {missing_label[:10]}")
        raise ValueError("\n".join(msg))

    keep_ids = [
        x for x in target_ids
        if x in expr_ids and x in meta_ids and (label_df is None or x in label_ids)
    ]

    expr_cells = expr_cells[keep_ids].copy()
    meta_obs = meta_obs.loc[keep_ids].copy()
    meta_cells = meta_cells[[cid in set(keep_ids) for cid in meta_cells.obs["cell_id"]]].copy()
    meta_cells = meta_cells[keep_ids].copy()
    meta_cells.obs = meta_obs.copy()

    return expr_cells, meta_cells, keep_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--latent-key", type=str, default="X_maxfuse")
    parser.add_argument("--neighbors-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixed-cell-ids", type=str, default=None)
    parser.add_argument("--strict-fixed-set", action="store_true")

    parser.add_argument(
        "--azimuth-csv",
        type=str,
        default="/data5/zhangye/scMRDR/input/PBMC/raw_input/pbmc_azimuth_annotations.csv",
        help="Path to Azimuth annotation csv."
    )
    parser.add_argument(
        "--azimuth-label-col",
        type=str,
        default="predicted.celltype.l2",
        help="Which Azimuth label column to use as final cell_type."
    )

    args = parser.parse_args()

    ratio_dir = Path(args.ratio_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    adata_post = sc.read_h5ad(str(ratio_dir / "training_adata_post.h5ad"))
    pred = sc.read_h5ad(str(ratio_dir / "pred_rna_val.h5ad"))
    true_rna_val = sc.read_h5ad(str(ratio_dir / "true_rna_val.h5ad"))
    split_info = json.loads((ratio_dir / "split_info_used.json").read_text())

    if args.latent_key not in adata_post.obsm:
        raise KeyError(
            f"{args.latent_key!r} not found in adata_post.obsm. "
            f"Available keys: {list(adata_post.obsm.keys())}"
        )

    adata_post.obs["cell_id"] = adata_post.obs_names.astype(str)
    pred.obs["cell_id"] = pred.obs_names.astype(str)
    true_rna_val.obs["cell_id"] = true_rna_val.obs_names.astype(str)

    fixed_cell_ids = read_fixed_cell_ids(args.fixed_cell_ids)

    current_val_query_ids = unique_preserve_order([
        str(c) for c in split_info["val_query_atac_cells"]
    ])

    if fixed_cell_ids is not None:
        fixed_cell_ids = unique_preserve_order([str(c) for c in fixed_cell_ids])
        invalid_fixed_ids = [c for c in fixed_cell_ids if c not in current_val_query_ids]
        if invalid_fixed_ids:
            raise ValueError(
                "The provided --fixed-cell-ids contains cells that are not in "
                "split_info['val_query_atac_cells'] for this ratio.\n"
                f"Number of invalid fixed ids: {len(invalid_fixed_ids)}\n"
                f"First few invalid ids: {invalid_fixed_ids[:10]}"
            )
        target_ids = fixed_cell_ids
    else:
        target_ids = current_val_query_ids

    # labels already present in true_rna_val.obs
    existing_label_df = build_label_df_from_obs(true_rna_val.obs)

    # azimuth labels
    azimuth_label_df = load_azimuth_label_df(
        args.azimuth_csv,
        label_col=args.azimuth_label_col
    )

    # merge labels, prefer existing labels if present, fill cell_type from azimuth
    label_df = merge_label_dfs(existing_label_df, azimuth_label_df)

    print(f"[Azimuth] loaded labels from: {args.azimuth_csv}")
    print(f"[Azimuth] final cell_type column uses: {args.azimuth_label_col}")
    print(f"[Azimuth] number of labeled cells: {len(azimuth_label_df)}")

    if "modality" not in adata_post.obs.columns:
        raise KeyError("Expected 'modality' column in training_adata_post.obs for MaxFuse output.")

    # MaxFuse 的 training_adata_post 里 RNA+ATAC 都有，这里只取 ATAC 视图
    meta_mask = (
        adata_post.obs["cell_id"].isin(target_ids)
        & adata_post.obs["modality"].astype(str).eq("atac").to_numpy()
    )
    meta_cells = adata_post[meta_mask].copy()

    expr_cells = pred.copy()

    expr_cells, meta_cells, keep_ids = align_to_target_ids(
        expr_cells=expr_cells,
        meta_cells=meta_cells,
        label_df=label_df,
        target_ids=target_ids,
        strict=args.strict_fixed_set or (fixed_cell_ids is not None),
    )

    print(f"[maxfuse val_atac] current_val_query_ids: {len(current_val_query_ids)}")
    print(f"[maxfuse val_atac] target_ids: {len(target_ids)}")
    print(f"[maxfuse val_atac] keep_ids: {len(keep_ids)}")

    if meta_cells.n_obs == 0:
        raise ValueError("No cells selected.")

    latent = np.asarray(meta_cells.obsm[args.latent_key], dtype=float)

    emb_adata = ad.AnnData(
        X=np.zeros((meta_cells.n_obs, 1), dtype=np.float32),
        obs=meta_cells.obs.copy()
    )
    emb_adata.obsm["X_latent"] = latent
    sc.pp.neighbors(
        emb_adata,
        use_rep="X_latent",
        n_neighbors=min(args.neighbors_k, max(2, meta_cells.n_obs - 1))
    )
    sc.tl.umap(emb_adata, random_state=args.seed)
    umap = np.asarray(emb_adata.obsm["X_umap"], dtype=float)

    obs = meta_cells.obs.copy()
    if "cell_id" not in obs.columns:
        obs["cell_id"] = obs.index.astype(str)

    obs.index = obs["cell_id"].astype(str)
    obs.index.name = "cell_id"
    obs["monocle_input_source"] = "maxfuse_val_atac"

    if "cell_type" not in obs.columns:
        raise ValueError(
            "No 'cell_type' column found after merging annotations. "
            "Please check Azimuth csv and chosen --azimuth-label-col."
        )

    var = expr_cells.var.copy()
    var.index.name = "gene_short_name"
    if "gene_short_name" not in var.columns:
        var["gene_short_name"] = var.index.astype(str)

    expr = pick_count_layer(expr_cells)
    expr = np.asarray(expr, dtype=np.float32)

    if expr.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Expression cells ({expr.shape[0]}) and metadata cells ({obs.shape[0]}) do not match."
        )
    if expr.shape[1] != var.shape[0]:
        raise ValueError(
            f"Expression genes ({expr.shape[1]}) and gene metadata ({var.shape[0]}) do not match."
        )

    expr_gc = sparse.csr_matrix(expr.T)

    spio.mmwrite(str(outdir / "expr_gene_by_cell.mtx"), expr_gc)
    obs.to_csv(outdir / "cell_metadata.csv")
    var.to_csv(outdir / "gene_metadata.csv")

    pd.DataFrame(
        latent,
        index=obs.index,
        columns=[f"latent_{i+1}" for i in range(latent.shape[1])]
    ).to_csv(outdir / "latent_embedding.csv")

    pd.DataFrame(
        umap,
        index=obs.index,
        columns=["UMAP_1", "UMAP_2"]
    ).to_csv(outdir / "umap_embedding.csv")

    out_adata = ad.AnnData(X=expr, obs=obs.copy(), var=var.copy())
    out_adata.obsm["X_latent"] = latent
    out_adata.obsm["X_umap"] = umap
    out_adata.write_h5ad(str(outdir / "monocle3_input_cells.h5ad"))

    print("Saved Monocle3 input to:", outdir)
    print("Final number of cells:", obs.shape[0])
    print("Final obs columns:")
    print(obs.columns.tolist())


if __name__ == "__main__":
    main()