# scMRDR → Monocle3 downstream pipeline for T3 / T4

This pipeline connects your trained **scMRDR aligned embedding** to **Monocle3** for:

- **Task 3: trajectory inference**
- **Task 4: future state prediction**

and evaluates them using the same style of metrics you used in your earlier scVelo-based code:

- **T3**: `PairAcc`, `PairAUC`
- **T4**: `F1`, `AUROC`

## Files

- `prepare_monocle3_input_from_scmrdr.py`
- `run_monocle3_on_scmrdr.R`
- `evaluate_monocle3_t3_t4.py`

## Design choice

### T3
Monocle3 is used to learn a principal graph and pseudotime **on the aligned embedding**.

### T4
Monocle3 does **not** provide a scVelo-style directed velocity transition matrix by default.  
So this pipeline builds a **graph-based future-state proxy** from the learned principal graph:

1. orient the principal graph using the chosen root node(s)
2. identify leaf nodes as terminal states
3. annotate leaves by majority `future_label`
4. score each cell against downstream leaves by graph distance
5. convert those scores into class probabilities
6. evaluate with `F1` and `AUROC`

This keeps the evaluation logic close to your previous T4 setting, but it is still a **Monocle3 graph-based approximation**, not a velocity transition probability.

---

## Expected scMRDR inputs per ratio directory

Your ratio directory should already contain files like:

- `training_adata_post.h5ad`
- `pred_rna_val.h5ad`
- `pred_rna_all_nonrna.h5ad`
- `split_info_used.json`

---

## Step 1. Prepare Monocle3 input from scMRDR output

Example:

```bash
python prepare_monocle3_input_from_scmrdr.py \
  --ratio-dir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040 \
  --outdir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_input \
  --cell-set val_atac \
  --latent-key latent_shared
```

### `--cell-set` options

- `val_atac`: recommended for evaluating held-out ATAC query cells
- `all_atac`: use all ATAC cells, with predicted RNA as Monocle3 input
- `all_rna`: use RNA cells directly

Outputs include:

- `expr_gene_by_cell.mtx`
- `cell_metadata.csv`
- `gene_metadata.csv`
- `latent_embedding.csv`
- `umap_embedding.csv`
- `monocle3_input_cells.h5ad`

---

## Step 2. Run Monocle3

Example:

```bash
Rscript run_monocle3_on_scmrdr.R \
  --input-dir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_input \
  --outdir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_run \
  --label-key celltype \
  --stage-order-json '{"HSC":0,"LMPP":1,"CMP":1,"GMP":1,"MEP":1,"Prog DC":1,"Prog MK":1,"DC":2,"Granulocyte":2,"Erythrocyte":2,"Megakaryocyte":2,"Platelet":2,"Mast Cells":2}'
```

This script:

- creates a `cell_data_set`
- keeps Monocle3 counts
- injects your custom aligned UMAP / latent embedding
- clusters cells on the aligned embedding
- learns the principal graph
- chooses roots from the earliest stage labels
- computes pseudotime

Outputs include:

- `monocle3_cells.csv`
- `principal_graph_edges.csv`
- `principal_graph_vertices.csv`
- `leaf_nodes.txt`
- `root_nodes.txt`
- `monocle3_cds.rds`

---

## Step 3. Evaluate T3 / T4

Example:

```bash
python evaluate_monocle3_t3_t4.py \
  --monocle-dir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_run \
  --input-h5ad /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_input/monocle3_input_cells.h5ad \
  --outdir /data5/zhangye/scMRDR/output/BMMC_test/scMRDR_results/single_040/monocle3_eval \
  --label-key celltype \
  --stage-order-json '{"HSC":0,"LMPP":1,"CMP":1,"GMP":1,"MEP":1,"Prog DC":1,"Prog MK":1,"DC":2,"Granulocyte":2,"Erythrocyte":2,"Megakaryocyte":2,"Platelet":2,"Mast Cells":2}' \
  --future-map-json '{"HSC":null,"LMPP":"lymphoid_dc","Prog B":"lymphoid_dc","Prog DC":"lymphoid_dc","DC":"lymphoid_dc","CMP":"myeloid","GMP":"myeloid","Granulocyte":"myeloid","Mast Cells":"myeloid","MEP":"ery_platelet","Prog MK":"ery_platelet","Megakaryocyte":"ery_platelet","Platelet":"ery_platelet","Erythrocyte":"ery_platelet"}' \
  --future-label-key future_label
```

Outputs:

- `T3_metrics.csv`
- `T4_metrics.csv`
- `T4_probabilities.csv`
- `T4_leaf_class_map.csv`
- `monocle3_cells_with_eval.csv`

---

## Recommended usage for your setting

For your current scMRDR setup, I recommend:

- use `--cell-set val_atac`
- use `latent_shared` as the aligned representation
- use your existing **stage order mapping** for T3
- use your existing **future label mapping** for T4
- compare Monocle3-based T3/T4 across the same ratio loop you already used for T1/T2

---

## Important note

This pipeline is intentionally faithful to your earlier evaluation style, but:

- **T3** is a direct Monocle3 pseudotime evaluation
- **T4** is a graph-distance-based future-state estimate from the Monocle3 principal graph

So T4 is **not numerically identical** to scVelo’s velocity-graph evaluation. It is the cleanest Monocle3-consistent analogue when you want the downstream model itself to be Monocle3.
