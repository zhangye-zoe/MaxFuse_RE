#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize


def load_json_maybe(x: str | None):
    if x is None:
        return None
    x = str(x)
    p = Path(x)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(x)


def _find_first_existing_key(container, candidates, kind="key"):
    for k in candidates:
        if k in container:
            return k
    raise KeyError(f"Could not find any valid {kind}. Tried: {candidates}")


def _resolve_label_key(df: pd.DataFrame, preferred=None):
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["celltype", "cell_type", "annotation", "annot", "labels", "leiden"])
    return _find_first_existing_key(df.columns, candidates, kind="label key")


def interclass_pairwise_accuracy(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    correct = 0.0
    total = 0.0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if ref[i] == ref[j]:
                continue
            w = abs(ref[i] - ref[j]) if weighted else 1.0
            if ref[i] < ref[j]:
                is_correct = pred[i] < pred[j]
            else:
                is_correct = pred[j] < pred[i]
            correct += w * float(is_correct)
            total += w
    return correct / total if total > 0 else np.nan


def interclass_pairwise_auc(pred, ref, weighted=True):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)

    valid = np.isfinite(pred) & np.isfinite(ref)
    pred = pred[valid]
    ref = ref[valid]

    classes = np.sort(np.unique(ref))
    aucs, weights = [], []

    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            mask = (ref == a) | (ref == b)
            y = (ref[mask] == b).astype(int)
            s = pred[mask]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, s)
            w = abs(b - a) if weighted else 1.0
            aucs.append(auc)
            weights.append(w)

    if len(aucs) == 0:
        return np.nan
    return float(np.average(np.asarray(aucs), weights=np.asarray(weights)))


def compute_t3_metrics(df, pred_time_key="pseudotime", reference_time_key=None,
                       label_key=None, stage_order_map=None, eval_mask=None):
    pred = pd.to_numeric(df[pred_time_key], errors="coerce").to_numpy()

    if eval_mask is None:
        eval_mask = np.ones(len(df), dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask).astype(bool)

    if reference_time_key is not None:
        ref = pd.to_numeric(df[reference_time_key], errors="coerce").to_numpy()
        reference_name = reference_time_key
    else:
        if label_key is None:
            label_key = _resolve_label_key(df)
        if stage_order_map is None:
            raise ValueError("stage_order_map must be provided when reference_time_key is None.")
        labels = df[label_key].astype(str)
        ref = labels.map(stage_order_map).to_numpy(dtype=float)
        reference_name = f"ordinal({label_key})"

    valid = eval_mask & np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 3:
        return {
            "PairAcc": np.nan,
            "PairAUC": np.nan,
            "n_eval": int(valid.sum()),
            "pred_time_key": pred_time_key,
            "reference": reference_name,
        }

    pred_v = pred[valid]
    ref_v = ref[valid]
    return {
        "PairAcc": float(interclass_pairwise_accuracy(pred_v, ref_v, weighted=True)),
        "PairAUC": float(interclass_pairwise_auc(pred_v, ref_v, weighted=True)),
        "n_eval": int(valid.sum()),
        "pred_time_key": pred_time_key,
        "reference": reference_name,
    }


def normalize_future_labels(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    return s


def build_leaf_class_map(cell_df: pd.DataFrame, leaf_nodes: list[str], lineage_key: str) -> dict[str, str]:
    tmp = cell_df.copy()
    tmp[lineage_key] = tmp[lineage_key].map(normalize_future_labels)
    tmp = tmp[tmp["closest_vertex"].isin(leaf_nodes) & tmp[lineage_key].notna()].copy()

    leaf_class_map = {}
    for leaf, g in tmp.groupby("closest_vertex"):
        votes = Counter(g[lineage_key].astype(str))
        leaf_class_map[leaf] = votes.most_common(1)[0][0]
    return leaf_class_map


def graph_distance_dict(g: nx.Graph, source: str) -> dict[str, float]:
    return nx.single_source_dijkstra_path_length(g, source=source, weight="weight")


def compute_monocle_future_probabilities(
    cell_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    root_nodes: list[str],
    lineage_key: str = "future_label",
    temperature: float = 1.0,
):
    g = nx.Graph()
    for _, row in edge_df.iterrows():
        g.add_edge(str(row["from"]), str(row["to"]), weight=1.0)

    deg = dict(g.degree())
    leaf_nodes = [n for n, d in deg.items() if d == 1 and n not in set(root_nodes)]

    if len(root_nodes) == 0:
        raise ValueError("No root nodes were provided; cannot orient the graph for T4.")
    if len(leaf_nodes) == 0:
        raise ValueError("No leaf nodes found in principal graph.")

    root_dist = {}
    for n in g.nodes():
        vals = []
        for r in root_nodes:
            try:
                vals.append(nx.shortest_path_length(g, r, n))
            except nx.NetworkXNoPath:
                pass
        root_dist[n] = min(vals) if len(vals) > 0 else np.inf

    leaf_class_map = build_leaf_class_map(cell_df, leaf_nodes, lineage_key=lineage_key)
    valid_leaf_nodes = [n for n in leaf_nodes if n in leaf_class_map]
    if len(valid_leaf_nodes) == 0:
        raise ValueError(
            "None of the graph leaves could be annotated from future labels. "
            "Check lineage_key / future_map / cell labels."
        )

    class_order = sorted(pd.unique([leaf_class_map[n] for n in valid_leaf_nodes]).tolist())

    dist_cache = {leaf: graph_distance_dict(g, leaf) for leaf in valid_leaf_nodes}
    scores = np.zeros((cell_df.shape[0], len(class_order)), dtype=float)
    per_leaf_scores = defaultdict(list)

    for i, (_, row) in enumerate(cell_df.iterrows()):
        v = str(row["closest_vertex"])
        if v not in g:
            continue

        downstream = [
            leaf for leaf in valid_leaf_nodes
            if np.isfinite(root_dist.get(v, np.inf))
            and np.isfinite(root_dist.get(leaf, np.inf))
            and root_dist[leaf] > root_dist[v]
        ]
        candidates = downstream if len(downstream) > 0 else valid_leaf_nodes

        leaf_scores = {}
        for leaf in candidates:
            d = dist_cache[leaf].get(v, np.inf)
            if not np.isfinite(d):
                continue
            leaf_scores[leaf] = np.exp(-d / max(temperature, 1e-8))

        if len(leaf_scores) == 0:
            continue

        class_score = {c: 0.0 for c in class_order}
        for leaf, s in leaf_scores.items():
            cls = leaf_class_map[leaf]
            class_score[cls] += s

        vec = np.array([class_score[c] for c in class_order], dtype=float)
        if vec.sum() > 0:
            vec = vec / vec.sum()
        scores[i] = vec

        for leaf, s in leaf_scores.items():
            per_leaf_scores["cell_id"].append(row["cell_id"])
            per_leaf_scores["leaf_node"].append(leaf)
            per_leaf_scores["leaf_class"].append(leaf_class_map[leaf])
            per_leaf_scores["score"].append(s)

    proba_df = pd.DataFrame(scores, index=cell_df["cell_id"], columns=[f"prob_{c}" for c in class_order])
    proba_df.insert(0, "cell_id", cell_df["cell_id"].to_list())

    leaf_meta = pd.DataFrame({
        "leaf_node": valid_leaf_nodes,
        "leaf_class": [leaf_class_map[n] for n in valid_leaf_nodes],
        "root_distance": [root_dist[n] for n in valid_leaf_nodes],
    })

    return proba_df, class_order, leaf_meta, pd.DataFrame(per_leaf_scores)


def compute_t4_metrics(cell_df: pd.DataFrame, proba_df: pd.DataFrame, class_order: list[str],
                       lineage_key="future_label", eval_mask=None):
    y_true_str = cell_df[lineage_key].map(normalize_future_labels).to_numpy()

    if eval_mask is None:
        eval_mask = np.ones(len(cell_df), dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask).astype(bool)

    valid_classes = np.isin(y_true_str, class_order)
    valid = eval_mask & valid_classes

    if valid.sum() < 3:
        return {
            "F1": np.nan,
            "AUROC": np.nan,
            "n_eval": int(valid.sum()),
            "class_order": class_order,
        }

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_true = np.array([class_to_idx[x] for x in y_true_str[valid]], dtype=int)

    score_cols = [f"prob_{c}" for c in class_order]
    y_score = proba_df.loc[valid, score_cols].to_numpy(dtype=float)
    y_pred = y_score.argmax(axis=1)

    f1 = f1_score(y_true, y_pred, average="macro")
    if len(class_order) == 2:
        auc = roc_auc_score(y_true, y_score[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=np.arange(len(class_order)))
        auc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")

    return {
        "F1": float(f1),
        "AUROC": float(auc),
        "n_eval": int(valid.sum()),
        "class_order": class_order,
    }


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate Monocle3 T3/T4 on scMRDR-aligned embeddings.")
    p.add_argument("--monocle-dir", type=str, required=True,
                   help="Directory containing monocle3_cells.csv and principal_graph_edges.csv")
    p.add_argument("--input-h5ad", type=str, required=True,
                   help="monocle3_input_cells.h5ad created by prepare_monocle3_input_from_scmrdr.py")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--label-key", type=str, default=None)
    p.add_argument("--reference-time-key", type=str, default=None)
    p.add_argument("--stage-order-json", type=str, default=None,
                   help="JSON string or JSON file path mapping label -> ordinal stage")
    p.add_argument("--future-map-json", type=str, default=None,
                   help="JSON string or JSON file path mapping label -> future lineage class")
    p.add_argument("--future-label-key", type=str, default="future_label")
    p.add_argument("--temperature", type=float, default=1.0)
    return p


def main():
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input_h5ad)
    cell_df = pd.read_csv(Path(args.monocle_dir) / "monocle3_cells.csv")
    edge_df = pd.read_csv(Path(args.monocle_dir) / "principal_graph_edges.csv")

    if "cell_id" not in cell_df.columns:
        raise ValueError("monocle3_cells.csv must contain cell_id")

    # keep only cells present in h5ad, and align order
    obs = adata.obs.copy()
    obs["cell_id"] = obs.index.astype(str)
    obs = obs.reset_index(drop=True)
    cell_df = cell_df.merge(obs, on="cell_id", how="left", suffixes=("", "_obs"))
    cell_df = cell_df[cell_df["cell_id"].isin(obs["cell_id"])].copy()
    cell_df = cell_df.set_index("cell_id").loc[obs["cell_id"]].reset_index()

    label_key = args.label_key or _resolve_label_key(cell_df)

    stage_order_map = load_json_maybe(args.stage_order_json)
    future_map = load_json_maybe(args.future_map_json)

    if args.future_label_key not in cell_df.columns:
        if future_map is None:
            raise ValueError(
                f"{args.future_label_key!r} not found in cell dataframe and no --future-map-json was provided."
            )
        cell_df[args.future_label_key] = cell_df[label_key].astype(str).map(future_map)

    with open(Path(args.monocle_dir) / "root_nodes.txt", "r", encoding="utf-8") as f:
        root_nodes = [x.strip() for x in f if x.strip()]

    t3 = compute_t3_metrics(
        df=cell_df,
        pred_time_key="pseudotime",
        reference_time_key=args.reference_time_key,
        label_key=label_key,
        stage_order_map=stage_order_map,
        eval_mask=np.isfinite(pd.to_numeric(cell_df["pseudotime"], errors="coerce").to_numpy())
    )
    pd.DataFrame([t3]).to_csv(outdir / "T3_metrics.csv", index=False)

    proba_df, class_order, leaf_meta, per_leaf_scores = compute_monocle_future_probabilities(
        cell_df=cell_df,
        edge_df=edge_df,
        root_nodes=root_nodes,
        lineage_key=args.future_label_key,
        temperature=args.temperature,
    )
    proba_df.to_csv(outdir / "T4_probabilities.csv", index=False)
    leaf_meta.to_csv(outdir / "T4_leaf_class_map.csv", index=False)
    per_leaf_scores.to_csv(outdir / "T4_leaf_scores_long.csv", index=False)

    t4 = compute_t4_metrics(
        cell_df=cell_df,
        proba_df=proba_df,
        class_order=class_order,
        lineage_key=args.future_label_key,
        eval_mask=np.isfinite(pd.to_numeric(cell_df["pseudotime"], errors="coerce").to_numpy())
    )
    pd.DataFrame([t4]).to_csv(outdir / "T4_metrics.csv", index=False)

    proba_df = proba_df.reset_index(drop=True)
    merged = cell_df.merge(proba_df, on="cell_id", how="left")
    merged.to_csv(outdir / "monocle3_cells_with_eval.csv", index=False)

    summary = {
        "label_key": label_key,
        "reference_time_key": args.reference_time_key,
        "future_label_key": args.future_label_key,
        "n_cells": int(cell_df.shape[0]),
        "t3": t3,
        "t4": t4,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
