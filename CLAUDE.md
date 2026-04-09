# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **patent thicket detection pipeline** that identifies dense webs of overlapping intellectual property rights as an unsupervised heterogeneous graph clustering problem. It combines PatentSBERTa embeddings with Heterogeneous GNNs and DMoN-DPR (Deep Modularity Networks with Diversity-Preserving Regularization).

## Running the Pipeline

**Primary entry point (recommended):**
```bash
python pipeline-live.py
# Interactive prompts for: technology name, K (clusters), W_dist, W_var
# Skips preprocessing/embedding/graph steps if cached data exists
```

**Web UI (Flask with live streaming):**
```bash
python server.py
# Serves on http://localhost:5000
```

**Docker:**
```bash
docker-compose up --build
```

**Batch execution (new timestamped results dir each run):**
```bash
python pipeline.py
```

## Architecture & Data Flow

```
raw-inputs/*.jsonl
    → [1] preprocess.py      → data/processed_patents.csv, citation_edges.csv, ownership_edges.csv
    → [2] embed.py           → data/patent_embeddings.npy       (PatentSBERTa, 768-dim)
    → [3] build_graph.py     → data/hetero_graph.pt             (PyG HeteroData)
    → [4] train.py           → models/dmon_dpr_model_{run_id}.pth
    → [5] analyze.py         → data/final_clusters_{run_id}.csv, thicket_stats_{run_id}.csv
    → [6] find_triples.py    → data/triple_stats_{run_id}.csv
```

**pipeline-live.py** caches steps 1–3 across runs (embeddings are the bottleneck). Steps 4–6 re-run for each hyperparameter combination. `run_id` format: `K{K}_Wd{W_dist}_Wv{W_var}`.

## Module Responsibilities (`src/`)

| Module | Key Function | Role |
|--------|-------------|------|
| `model_dmon.py` | `DMON_DPR_Model` | GCNConv (768→512) + linear pooling (512→K); outputs soft cluster assignments |
| `preprocess.py` | `preprocess_patents()` | Parses Lens.org JSONL; extracts English patents, citations, ownership into 3 CSVs |
| `embed.py` | `generate_embeddings()` | Runs PatentSBERTa on title+abstract; auto-detects CUDA/MPS/CPU |
| `build_graph.py` | `build_hetero_graph()` | Creates HeteroData with patent nodes (768-dim), firm nodes (1-dim), cites/owns edges |
| `train.py` | `train_dmon_dpr()` | 200 epochs Adam; loss = modularity + collapse + W_dist·distance + W_var·variance + entropy |
| `analyze.py` | `run_analysis()` | Computes Clarkson density, fragmentation index, internal citations per cluster |
| `find_triples.py` | `find_thicket_triples()` | Finds cyclic A→B→C→A inter-firm citation patterns within clusters |

## Key Design Decisions

**Loss function** in `train.py` combines five terms:
- `L_modularity`: rewards dense intra-cluster, sparse inter-cluster edges
- `L_collapse`: prevents all patents collapsing to one cluster
- `L_distance`: penalizes clusters with centroids closer than epsilon (semantic separation)
- `L_variance`: encourages balanced assignment dispersion
- `L_entropy`: pushes soft assignments toward crisp (confident) decisions

**Graph structure** is bipartite heterogeneous: patent nodes carry 768-dim PatentSBERTa features; firm nodes have 1-dim placeholder features. Two edge types: `(patent, cites, patent)` and `(firm, owns, patent)`. Citation edges are temporally filtered to a 0–5 year window.

**Thicket candidates** in `analyze.py` are clusters with < 500 patents AND > 5 internal citations. Metrics: Clarkson density = internal_citations / (n*(n-1)/2), fragmentation index = unique_firms / cluster_size.

## Input Data Format

Raw input is Lens.org JSONL exports in `raw-inputs/`. The preprocessing step expects:
- `biblio.invention_title[*].{text, lang}` for title
- `biblio.abstract[*].{text, lang}` for abstract
- `biblio.parties.applicants[*].extracted_name.value` for firm names
- `biblio.references_cited.citations` and `biblio.cited_by.patents` for citation edges

## Output Structure

```
results/{tech_name}/
├── data/               # CSVs, .npy, .pt, cluster outputs
├── models/             # Trained .pth files per run_id
└── pipeline_log_{run_id}.txt
```

## Setup

```bash
uv sync          # creates .venv and installs all deps (one-time)
```

Python 3.10 (`uv` reads `.python-version` automatically). Core deps: `torch`, `torch_geometric`, `sentence_transformers`, `flask`, `networkx`, `pandas`, `scikit-learn`. See `pyproject.toml`.
