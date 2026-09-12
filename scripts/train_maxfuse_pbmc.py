#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from maxfuse.model import Fusor

# ==========================================================
# Paths
# ==========================================================
INPUT_DIR = "/data5/zhangye/scMRDR/input/PBMC/preprocessed_input"
OUTPUT_DIR = "/data5/zhangye/maxfuse/output/PBMC"
SPLIT_ROOT = os.path.join(INPUT_DIR, "results_ratio_loop")
ATAC_GAS_PATH = os.path.join(INPUT_DIR, "ATAC_gas.h5ad")
OUT_ROOT = os.path.join(OUTPUT_DIR, "MaxFuse_results")

RATIO_LABELS = [f"single_{x:03d}" for x in [00, 20, 40, 60, 80, 100]]
SEED = 1234

# ==========================================================
# General config
# ==========================================================
MIN_COMMON_FEATURES = 50
N_PCS_EVAL = 30
INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA = True
USE_LAYER_COUNTS_FOR_RNA_PREDICTION = True

# ==========================================================
# MaxFuse hyperparameters
# ==========================================================
MAXFUSE_METHOD = "centroid_shrinkage"  # or "graph_smoothing"

# batching
MAX_OUTWARD_SIZE = 5000
MATCHING_RATIO = 5
METACELL_SIZE = 2
BATCH_SPLIT_METHOD = "random"  # or "binning"
BATCHING_SCHEME = "pairwise"
PREBATCHING_SMOOTHING = False

# graph construction
N_NEIGHBORS1 = 15
N_NEIGHBORS2 = 15
GRAPH_SVD_COMPONENTS1 = 30
GRAPH_SVD_COMPONENTS2 = 30
RESOLUTION1 = 1.0
RESOLUTION2 = 1.0
GRAPH_METRIC = "correlation"
LEIDEN_RUNS = 1

# initial pivots
INIT_WT1 = 0.3
INIT_WT2 = 0.3
INIT_SVD_COMPONENTS1 = 30
INIT_SVD_COMPONENTS2 = 30

# refined pivots
REFINE_WT1 = 0.5
REFINE_WT2 = 0.5
REFINE_SVD_COMPONENTS1 = 50
REFINE_SVD_COMPONENTS2 = 50
CCA_COMPONENTS = 20
REFINE_FILTER_PROP = 0.0
REFINE_ITERS = 2
CCA_MAX_ITER = 2000

# pivot filtering + propagation
FILTER_BAD_MATCHES_PROP = 0.0
PROP_WT1 = 0.7
PROP_WT2 = 0.7
PROP_SVD_COMPONENTS1 = 30
PROP_SVD_COMPONENTS2 = 30
PROP_METRIC = "euclidean"

# prediction / embedding
MATCH_ORDER_FOR_ATAC_QUERY = (2, 1)  # guarantee each arr2 (ATAC) has at least one RNA match
EMBED_REFIT = True
EMBED_CCA_COMPONENTS = CCA_COMPONENTS


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def safe_cor(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1 or y.size <= 1:
        return np.nan
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def hit_at_k(cross_dist: np.ndarray, k: int = 5) -> float:
    nq = cross_dist.shape[0]
    hits = []
    for i in range(nq):
        ord_idx = np.argsort(cross_dist[i, :])[: min(k, cross_dist.shape[1])]
        hits.append(i in ord_idx)
    return float(np.mean(hits)) if len(hits) > 0 else np.nan


def sanitize_adata_for_write(adata):
    adata = adata.copy()
    if isinstance(adata.X, np.matrix):
        adata.X = np.asarray(adata.X)
    for k in list(adata.layers.keys()):
        if isinstance(adata.layers[k], np.matrix):
            adata.layers[k] = np.asarray(adata.layers[k])
    return adata


def save_h5ad_safe(adata, path):
    sanitize_adata_for_write(adata).write(str(path))


def subset_adata_by_cells(adata, cells):
    cells = [c for c in cells if c in adata.obs_names]
    return adata[cells].copy()


def get_common_names(*arrays):
    if len(arrays) == 0:
        return []
    common = set(arrays[0])
    for arr in arrays[1:]:
        common &= set(arr)
    return sorted(common)



def get_rna_matrix_for_prediction(adata_rna: ad.AnnData) -> np.ndarray:
    if USE_LAYER_COUNTS_FOR_RNA_PREDICTION and "counts" in adata_rna.layers:
        return to_dense(adata_rna.layers["counts"])
    return to_dense(adata_rna.X)



def build_model_input_for_ratio(split_dir, atac_gas_global):
    split_dir = Path(split_dir)

    train_rna_ref = sc.read_h5ad(str(split_dir / "train_rna_ref.h5ad"))
    val_true_rna = sc.read_h5ad(str(split_dir / "val_true_rna.h5ad"))
    val_atac_activity = sc.read_h5ad(str(split_dir / "val_atac_activity.h5ad"))

    with open(split_dir / "split_info.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_cells = split_info["train_cells"]
    val_query_atac_cells = split_info["val_query_atac_cells"]

    train_atac_activity = subset_adata_by_cells(atac_gas_global, train_cells)

    common_features = get_common_names(
        train_rna_ref.var_names.tolist(),
        train_atac_activity.var_names.tolist(),
        val_atac_activity.var_names.tolist(),
        val_true_rna.var_names.tolist(),
    )
    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features in {split_dir.name}: {len(common_features)}")

    train_rna_ref = train_rna_ref[:, common_features].copy()
    train_atac_activity = train_atac_activity[:, common_features].copy()
    val_atac_activity = val_atac_activity[:, common_features].copy()
    val_true_rna = val_true_rna[:, common_features].copy()

    train_rna_ref.obs["modality"] = "rna"
    train_atac_activity.obs["modality"] = "atac"
    val_atac_activity.obs["modality"] = "atac"
    val_true_rna.obs["modality"] = "rna"

    for adata_obj in [train_rna_ref, train_atac_activity, val_atac_activity, val_true_rna]:
        if "batch" not in adata_obj.obs.columns:
            adata_obj.obs["batch"] = "batch0"

    if "counts" in train_rna_ref.layers:
        train_rna_ref.layers["count"] = train_rna_ref.layers["counts"]
    else:
        train_rna_ref.layers["count"] = train_rna_ref.X.copy()

    if "counts" in train_atac_activity.layers:
        train_atac_activity.layers["count"] = train_atac_activity.layers["counts"]
    else:
        train_atac_activity.layers["count"] = train_atac_activity.X.copy()

    if "counts" in val_atac_activity.layers:
        val_atac_activity.layers["count"] = val_atac_activity.layers["counts"]
    else:
        val_atac_activity.layers["count"] = val_atac_activity.X.copy()

    if INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA:
        adata_model = ad.concat(
            {
                "train_rna": train_rna_ref.copy(),
                "train_atac": train_atac_activity.copy(),
                "val_atac": val_atac_activity.copy(),
            },
            axis=0,
            join="inner",
            label="dataset_block",
            index_unique=None,
        )
    else:
        adata_model = ad.concat(
            {
                "train_rna": train_rna_ref.copy(),
                "train_atac": train_atac_activity.copy(),
            },
            axis=0,
            join="inner",
            label="dataset_block",
            index_unique=None,
        )

    adata_model.obs["is_val_query"] = adata_model.obs_names.isin(val_query_atac_cells)

    return {
        "train_rna_ref": train_rna_ref,
        "train_atac_activity": train_atac_activity,
        "val_atac_activity": val_atac_activity,
        "val_true_rna": val_true_rna,
        "adata_model": adata_model,
        "split_info": split_info,
        "common_features": common_features,
    }



def build_maxfuse_arrays(train_rna_ref, train_atac_activity, val_atac_activity):
    if INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA:
        atac_all = ad.concat(
            {
                "train_atac": train_atac_activity.copy(),
                "val_atac": val_atac_activity.copy(),
            },
            axis=0,
            join="inner",
            label="dataset_block",
            index_unique=None,
        )
    else:
        atac_all = train_atac_activity.copy()

    shared_arr1 = to_dense(train_rna_ref.X)
    shared_arr2 = to_dense(atac_all.X)

    active_arr1 = to_dense(train_rna_ref.X)
    active_arr2 = to_dense(atac_all.X)

    return atac_all, shared_arr1, shared_arr2, active_arr1, active_arr2



def run_maxfuse(train_rna_ref, train_atac_activity, val_atac_activity):
    atac_all, shared_arr1, shared_arr2, active_arr1, active_arr2 = build_maxfuse_arrays(
        train_rna_ref=train_rna_ref,
        train_atac_activity=train_atac_activity,
        val_atac_activity=val_atac_activity,
    )

    fusor = Fusor(
        shared_arr1=shared_arr1,
        shared_arr2=shared_arr2,
        active_arr1=active_arr1,
        active_arr2=active_arr2,
        method=MAXFUSE_METHOD,
        labels1=None,
        labels2=None,
    )

    print("[1/7] split_into_batches", flush=True)
    fusor.split_into_batches(
        max_outward_size=MAX_OUTWARD_SIZE,
        matching_ratio=MATCHING_RATIO,
        metacell_size=METACELL_SIZE,
        method=BATCH_SPLIT_METHOD,
        batching_scheme=BATCHING_SCHEME,
        prebatching_smoothing=PREBATCHING_SMOOTHING,
        seed=SEED,
        verbose=True,
    )

    print("[2/7] construct_graphs", flush=True)
    fusor.construct_graphs(
        n_neighbors1=N_NEIGHBORS1,
        n_neighbors2=N_NEIGHBORS2,
        svd_components1=GRAPH_SVD_COMPONENTS1,
        svd_components2=GRAPH_SVD_COMPONENTS2,
        resolution1=RESOLUTION1,
        resolution2=RESOLUTION2,
        randomized_svd=False,
        svd_runs=1,
        leiden_runs=LEIDEN_RUNS,
        metric=GRAPH_METRIC,
        leiden_seed=SEED,
        verbose=True,
    )

    print("[3/7] find_initial_pivots", flush=True)
    fusor.find_initial_pivots(
        wt1=INIT_WT1,
        wt2=INIT_WT2,
        svd_components1=INIT_SVD_COMPONENTS1,
        svd_components2=INIT_SVD_COMPONENTS2,
        randomized_svd=False,
        svd_runs=1,
        verbose=True,
    )

    print("[4/7] refine_pivots", flush=True)
    fusor.refine_pivots(
        wt1=REFINE_WT1,
        wt2=REFINE_WT2,
        svd_components1=REFINE_SVD_COMPONENTS1,
        svd_components2=REFINE_SVD_COMPONENTS2,
        cca_components=CCA_COMPONENTS,
        filter_prop=REFINE_FILTER_PROP,
        n_iters=REFINE_ITERS,
        randomized_svd=False,
        svd_runs=1,
        cca_max_iter=CCA_MAX_ITER,
        verbose=True,
    )

    print("[5/7] filter_bad_matches(pivot)", flush=True)
    fusor.filter_bad_matches(
        target="pivot",
        filter_prop=FILTER_BAD_MATCHES_PROP,
        verbose=True,
    )

    print("[6/7] propagate", flush=True)
    fusor.propagate(
        wt1=PROP_WT1,
        wt2=PROP_WT2,
        svd_components1=PROP_SVD_COMPONENTS1,
        svd_components2=PROP_SVD_COMPONENTS2,
        metric=PROP_METRIC,
        randomized_svd=False,
        svd_runs=1,
        verbose=True,
    )

    print("[7/7] filter_bad_matches(propagated)", flush=True)
    fusor.filter_bad_matches(
        target="propagated",
        filter_prop=0.0,
        verbose=True,
    )

    full_matching = fusor.get_matching(order=MATCH_ORDER_FOR_ATAC_QUERY, target="full_data")
    emb_rna, emb_atac = fusor.get_embedding(
        active_arr1=active_arr1,
        active_arr2=active_arr2,
        refit=EMBED_REFIT,
        matching=full_matching,
        order=MATCH_ORDER_FOR_ATAC_QUERY,
        cca_components=EMBED_CCA_COMPONENTS,
        cca_max_iter=CCA_MAX_ITER,
    )

    return fusor, atac_all, full_matching, emb_rna, emb_atac



def predict_rna_from_matching(train_rna_ref, atac_all, full_matching):
    rows, cols, scores = full_matching
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    scores = np.asarray(scores, dtype=float)

    idx2_to_idx1 = defaultdict(list)
    for i1, i2, sc_ in zip(rows, cols, scores):
        idx2_to_idx1[i2].append((i1, sc_))

    rna_mat = get_rna_matrix_for_prediction(train_rna_ref)
    pred = np.zeros((atac_all.n_obs, train_rna_ref.n_vars), dtype=float)
    match_n = np.zeros(atac_all.n_obs, dtype=int)

    for i2 in range(atac_all.n_obs):
        matched = idx2_to_idx1.get(i2, [])
        if len(matched) == 0:
            pred[i2, :] = np.nan
            continue

        idx1 = np.array([x[0] for x in matched], dtype=int)
        wt = np.array([x[1] for x in matched], dtype=float)
        match_n[i2] = len(idx1)

        wt = np.nan_to_num(wt, nan=0.0, posinf=0.0, neginf=0.0)
        # MaxFuse score is similarity-like but may contain negative values.
        # Use rank-preserving positive weights when possible; otherwise fall back to uniform averaging.
        wt = wt - np.min(wt)
        if np.allclose(wt.sum(), 0.0):
            wt = np.ones_like(wt) / len(wt)
        else:
            wt = wt / wt.sum()

        pred[i2, :] = np.average(rna_mat[idx1, :], axis=0, weights=wt)

    pred_adata = ad.AnnData(
        X=pred,
        obs=atac_all.obs.copy(),
        var=train_rna_ref.var.copy(),
    )
    pred_adata.obs_names = atac_all.obs_names.copy()
    pred_adata.var_names = train_rna_ref.var_names.copy()
    pred_adata.obs["modality"] = "pred_rna_from_atac"
    pred_adata.obs["n_matched_rna"] = match_n
    pred_adata.uns["prediction_method"] = "MaxFuse_full_matching_weighted_average"
    return pred_adata



def build_training_adata_post(train_rna_ref, atac_all, emb_rna, emb_atac):
    rna_post = train_rna_ref.copy()
    atac_post = atac_all.copy()
    rna_post.obsm["X_maxfuse"] = np.asarray(emb_rna)
    atac_post.obsm["X_maxfuse"] = np.asarray(emb_atac)

    adata_post = ad.concat(
        {
            "train_rna": rna_post,
            "atac_all": atac_post,
        },
        axis=0,
        join="inner",
        label="dataset_block",
        index_unique=None,
    )
    return adata_post



def evaluate_t2(pred_rna_val, true_rna_val, outdir):
    common_features = get_common_names(pred_rna_val.var_names.tolist(), true_rna_val.var_names.tolist())
    common_cells = get_common_names(pred_rna_val.obs_names.tolist(), true_rna_val.obs_names.tolist())

    if len(common_features) < MIN_COMMON_FEATURES:
        raise ValueError(f"Too few common features for T2 evaluation: {len(common_features)}")
    if len(common_cells) == 0:
        raise ValueError("No common cells for T2 evaluation.")

    pred = pred_rna_val[common_cells, common_features].copy()
    true = true_rna_val[common_cells, common_features].copy()

    pred_mat = to_dense(pred.X).T
    true_mat = to_dense(true.X).T

    cell_cor = np.array([safe_cor(pred_mat[:, i], true_mat[:, i]) for i in range(pred_mat.shape[1])], dtype=float)
    gene_cor = np.array([safe_cor(pred_mat[i, :], true_mat[i, :]) for i in range(pred_mat.shape[0])], dtype=float)

    mse = float(np.nanmean((pred_mat - true_mat) ** 2))
    rmse = float(np.sqrt(mse))

    t2_metrics = pd.DataFrame({
        "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
        "value": [float(np.nanmean(cell_cor)), float(np.nanmean(gene_cor)), rmse],
    })
    t2_metrics.to_csv(Path(outdir) / "T2_metrics.csv", index=False)

    pd.DataFrame({"cell": common_cells, "cellwise_pearson": cell_cor}).to_csv(
        Path(outdir) / "T2_cellwise_pearson.csv", index=False
    )
    pd.DataFrame({"gene": common_features, "genewise_pearson": gene_cor}).to_csv(
        Path(outdir) / "T2_genewise_pearson.csv", index=False
    )

    return t2_metrics, pred_mat, true_mat, common_cells, common_features



def evaluate_t1_from_true_pred(pred_mat, true_mat, common_cells, common_features, outdir):
    true_cells_by_features = true_mat.T
    pred_cells_by_features = pred_mat.T
    mix = np.vstack([true_cells_by_features, pred_cells_by_features])

    n_components = min(N_PCS_EVAL, mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("PCA components < 2, cannot evaluate T1.")

    pca = PCA(n_components=n_components, random_state=SEED)
    emb = pca.fit_transform(mix)

    n_q = true_cells_by_features.shape[0]
    emb_true = emb[:n_q, :]
    emb_pred = emb[n_q:, :]

    dist_mat = pairwise_distances(np.vstack([emb_true, emb_pred]), metric="euclidean")
    cross_dist = dist_mat[:n_q, n_q : (2 * n_q)]

    paired_dist = np.diag(cross_dist).astype(float)
    foscttm_each = np.array([np.mean(cross_dist[i, :] < paired_dist[i]) for i in range(n_q)], dtype=float)
    top1_acc = hit_at_k(cross_dist, k=1)
    top5_acc = hit_at_k(cross_dist, k=5)
    top10_acc = hit_at_k(cross_dist, k=10)

    nn_idx = np.argmin(cross_dist, axis=1)
    nn_match = (nn_idx == np.arange(n_q))

    t1_metrics = pd.DataFrame({
        "metric": [
            "paired_embedding_distance_mean",
            "paired_embedding_distance_median",
            "FOSCTTM",
            "Top1_ACC",
            "Top5_ACC",
            "Top10_ACC",
        ],
        "value": [
            float(np.nanmean(paired_dist)),
            float(np.nanmedian(paired_dist)),
            float(np.nanmean(foscttm_each)),
            top1_acc,
            top5_acc,
            top10_acc,
        ],
    })
    t1_metrics.to_csv(Path(outdir) / "T1_metrics.csv", index=False)

    pd.DataFrame({
        "cell": list(common_cells),
        "FOSCTTM": foscttm_each,
        "paired_dist": paired_dist,
        "Top1_match": nn_match.astype(bool),
    }).to_csv(Path(outdir) / "T1_per_cell_metrics.csv", index=False)

    pca_df = pd.DataFrame(emb[:, : min(5, emb.shape[1])], columns=[f"PC{i+1}" for i in range(min(5, emb.shape[1]))])
    pca_df["group"] = ["True_RNA"] * n_q + ["Pred_from_ATAC"] * n_q
    pca_df["cell"] = [f"true_{c}" for c in common_cells] + [f"pred_{c}" for c in common_cells]
    pca_df.to_csv(Path(outdir) / "pca_true_pred_coords.csv", index=False)

    return t1_metrics



def save_matching_csv(full_matching, train_rna_ref, atac_all, outdir):
    rows, cols, scores = full_matching
    df = pd.DataFrame({
        "rna_index": np.asarray(rows, dtype=int),
        "atac_index": np.asarray(cols, dtype=int),
        "score": np.asarray(scores, dtype=float),
    })
    df["rna_cell"] = train_rna_ref.obs_names.values[df["rna_index"].values]
    df["atac_cell"] = atac_all.obs_names.values[df["atac_index"].values]
    df.to_csv(Path(outdir) / "full_matching.csv", index=False)



def main():
    ensure_dir(OUT_ROOT)
    atac_gas_global = sc.read_h5ad(ATAC_GAS_PATH)

    all_summary = []

    for ratio_label in RATIO_LABELS:
        split_dir = Path(SPLIT_ROOT) / ratio_label
        outdir = Path(OUT_ROOT) / ratio_label
        ensure_dir(outdir)

        if not split_dir.exists():
            warnings.warn(f"Missing split directory: {split_dir}. Skipping.")
            continue

        print("\n============================", flush=True)
        print("Running ratio:", ratio_label, flush=True)
        print("============================", flush=True)

        try:
            bundle = build_model_input_for_ratio(split_dir, atac_gas_global)
            train_rna_ref = bundle["train_rna_ref"]
            train_atac_activity = bundle["train_atac_activity"]
            val_atac_activity = bundle["val_atac_activity"]
            true_rna_val = bundle["val_true_rna"]
            adata_model = bundle["adata_model"]
            split_info = bundle["split_info"]
            common_features = bundle["common_features"]

            save_h5ad_safe(adata_model, outdir / "training_adata_input.h5ad")

            fusor, atac_all, full_matching, emb_rna, emb_atac = run_maxfuse(
                train_rna_ref=train_rna_ref,
                train_atac_activity=train_atac_activity,
                val_atac_activity=val_atac_activity,
            )

            adata_post = build_training_adata_post(
                train_rna_ref=train_rna_ref,
                atac_all=atac_all,
                emb_rna=emb_rna,
                emb_atac=emb_atac,
            )
            save_h5ad_safe(adata_post, outdir / "training_adata_post.h5ad")

            pred_rna_all = predict_rna_from_matching(
                train_rna_ref=train_rna_ref,
                atac_all=atac_all,
                full_matching=full_matching,
            )
            pred_rna_all.obsm["X_maxfuse"] = np.asarray(emb_atac)
            save_h5ad_safe(pred_rna_all, outdir / "pred_rna_all_nonrna.h5ad")

            val_cells = split_info["val_query_atac_cells"]
            pred_rna_val = pred_rna_all[[c for c in val_cells if c in pred_rna_all.obs_names]].copy()

            save_h5ad_safe(pred_rna_val, outdir / "pred_rna_val.h5ad")
            save_h5ad_safe(true_rna_val, outdir / "true_rna_val.h5ad")
            save_matching_csv(full_matching, train_rna_ref, atac_all, outdir)

            t2_metrics, pred_mat, true_mat, common_cells, common_features_eval = evaluate_t2(
                pred_rna_val=pred_rna_val,
                true_rna_val=true_rna_val,
                outdir=outdir,
            )

            t1_metrics = evaluate_t1_from_true_pred(
                pred_mat=pred_mat,
                true_mat=true_mat,
                common_cells=common_cells,
                common_features=common_features_eval,
                outdir=outdir,
            )

            with open(outdir / "split_info_used.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)

            config_dump = {
                "MAXFUSE_METHOD": MAXFUSE_METHOD,
                "MAX_OUTWARD_SIZE": MAX_OUTWARD_SIZE,
                "MATCHING_RATIO": MATCHING_RATIO,
                "METACELL_SIZE": METACELL_SIZE,
                "BATCH_SPLIT_METHOD": BATCH_SPLIT_METHOD,
                "BATCHING_SCHEME": BATCHING_SCHEME,
                "N_NEIGHBORS1": N_NEIGHBORS1,
                "N_NEIGHBORS2": N_NEIGHBORS2,
                "GRAPH_SVD_COMPONENTS1": GRAPH_SVD_COMPONENTS1,
                "GRAPH_SVD_COMPONENTS2": GRAPH_SVD_COMPONENTS2,
                "INIT_WT1": INIT_WT1,
                "INIT_WT2": INIT_WT2,
                "REFINE_WT1": REFINE_WT1,
                "REFINE_WT2": REFINE_WT2,
                "CCA_COMPONENTS": CCA_COMPONENTS,
                "REFINE_ITERS": REFINE_ITERS,
                "PROP_WT1": PROP_WT1,
                "PROP_WT2": PROP_WT2,
                "MATCH_ORDER_FOR_ATAC_QUERY": list(MATCH_ORDER_FOR_ATAC_QUERY),
                "USE_LAYER_COUNTS_FOR_RNA_PREDICTION": USE_LAYER_COUNTS_FOR_RNA_PREDICTION,
                "INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA": INCLUDE_VAL_QUERY_ATAC_IN_MODEL_ADATA,
                "common_features_input": len(common_features),
                "common_features_eval": len(common_features_eval),
            }
            with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(config_dump, f, indent=2, ensure_ascii=False)

            t2_map = dict(zip(t2_metrics["metric"], t2_metrics["value"]))
            t1_map = dict(zip(t1_metrics["metric"], t1_metrics["value"]))

            all_summary.append({
                "ratio_label": ratio_label,
                "single_frac": split_info["single_frac"],
                "train_paired": len(split_info["train_paired_cells"]),
                "train_rna_only": len(split_info["train_rna_only_cells"]),
                "train_atac_only": len(split_info["train_atac_only_cells"]),
                "val_paired": len(split_info["val_paired_cells"]),
                "val_rna_only": len(split_info["val_rna_only_cells"]),
                "val_atac_only": len(split_info["val_atac_only_cells"]),
                "query_cells": len(common_cells),
                "common_features": len(common_features_eval),
                "cellwise_pearson_mean": t2_map["pearson_cell_mean"],
                "genewise_pearson_mean": t2_map["pearson_gene_mean"],
                "rmse": t2_map["rmse"],
                "paired_embedding_distance_mean": t1_map["paired_embedding_distance_mean"],
                "paired_embedding_distance_median": t1_map["paired_embedding_distance_median"],
                "foscttm": t1_map["FOSCTTM"],
                "top1_acc": t1_map["Top1_ACC"],
                "top5_acc": t1_map["Top5_ACC"],
                "top10_acc": t1_map["Top10_ACC"],
            })

            print("Finished:", ratio_label, flush=True)
            print(pd.DataFrame(all_summary).tail(1), flush=True)

        except Exception as e:
            warnings.warn(f"Failed on {ratio_label}: {repr(e)}")
            with open(outdir / "error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e) + "\n")
            continue

    if len(all_summary) > 0:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(Path(OUT_ROOT) / "summary_all_ratios_MaxFuse.csv", index=False)
        print("\nAll finished. Summary saved to:", flush=True)
        print(Path(OUT_ROOT) / "summary_all_ratios_MaxFuse.csv", flush=True)
        print(summary_df, flush=True)
    else:
        print("No successful ratio runs.", flush=True)


if __name__ == "__main__":
    main()
