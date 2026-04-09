# RUNBOOK — Patent Thicket Pipeline

## First-time setup

```bash
uv sync
```

This creates `.venv` and installs all dependencies. Re-run after pulling changes if `pyproject.toml` changes.

---

## Running the pipeline

```bash
uv run python pipeline-live.py
```

You will be prompted in this order:

| Prompt | What to enter | Notes |
|--------|--------------|-------|
| Technology name | e.g. `fcev-1998-2012` | Must match the filename of your `.jsonl` (without extension). Creates `results/{name}/` |
| Path to `.jsonl` | e.g. `raw-inputs/fcev-1998-2012.jsonl` | **Only asked on the first run** — skipped once embeddings are cached |
| `K` | number of clusters | 50–100 for exploration, 200–400 for production-level analysis |
| `W_dist` | semantic distance weight | `1.0` default; raise to `5.0–10.0` to force more semantically distinct clusters |
| `W_var` | assignment variance weight | `1.0` default; rarely needs tuning |

---

## Caching behaviour

Steps 1–3 (preprocess → embed → build graph) are **cached** inside `results/{tech}/data/` and only run once per technology. Steps 4–6 always re-run, identified by `run_id = K{K}_Wd{W_dist}_Wv{W_var}`.

- **To experiment with different hyperparameters**: just re-run `pipeline-live.py` — it jumps straight to step 4.
- **To start fully fresh**: delete `results/{tech}/data/patent_embeddings.npy` (or the entire `data/` folder).
- **Embedding time on M1**: roughly 20–40 min for a ~5k-patent dataset. Everything else is minutes.

---

## Output files

```
results/{tech_name}/
├── data/
│   ├── processed_patents.csv        patents with title, first_claim, abstract, date
│   ├── citation_edges.csv           (citing, cited) lens_id pairs
│   ├── ownership_edges.csv          (firm, patent_id) — firm names normalised
│   ├── patent_embeddings.npy        768-dim PatentSBERTa matrix            [cached]
│   ├── hetero_graph.pt              PyG HeteroData object                  [cached]
│   ├── final_clusters_{run_id}.csv  patent_id → cluster assignment
│   └── thicket_stats_{run_id}.csv   per-cluster density & fragmentation metrics
├── models/
│   └── dmon_dpr_model_{run_id}.pth  trained model weights
└── pipeline_log_{run_id}.txt        full stdout of the run
```

---

## Things to keep in mind

- **Directory naming**: raw data lives in `raw-inputs/` (hyphen). The `raw_inputs/` (underscore) folder at the root is a Docker volume mount — do not put your data there for local runs.
- **MPS fallback**: if you see an MPS-related crash during embedding, prefix the command with `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python pipeline-live.py`.
- **Family deduplication**: preprocessing keeps only one representative per simple patent family, so the output row count will be ~30% of the raw record count — this is expected and intentional.
- **Firm names**: normalised to uppercase with legal suffixes stripped (INC, LLC, CORP, KK, etc.). Two entries that differ only by suffix will be merged into one graph node.

---

## Web UI (optional)

```bash
uv run python server.py   # http://localhost:5000
```

Mirrors `pipeline-live.py` but renders live output in a browser terminal. Useful for demos.

---

## Docker (for sharing / reproducibility)

```bash
docker-compose up --build   # http://localhost:5000
```

The container mounts `raw-inputs/` → `/app/raw_inputs` and `results/` → `/app/results`, so outputs persist on your machine. Note: Docker has no MPS access, so the embedding step runs on CPU and will be significantly slower.
