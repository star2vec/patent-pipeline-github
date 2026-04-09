# Methodology Outline — Patent Thicket Detection Pipeline

---

## 3.1 Data Collection & Preprocessing (`preprocess.py`)

**Data source**
- Explain that data is sourced from Lens.org as JSONL exports
- State which fields are extracted: `biblio.invention_title`, `abstract`, `biblio.parties.applicants[*].extracted_name.value`, `biblio.references_cited.citations`, `biblio.cited_by.patents`, `date_published`, `claims`
- Mention the technology sectors chosen and the query logic used (IPC/CPC codes, date ranges) — **this is missing from your draft AND from the code; must be described manually based on your actual Lens.org query**

**Language filtering**
- Explain that only English-language titles and abstracts are retained (lang == 'en')
- Patents with no English abstract AND no English first claim are discarded entirely

**Simple family deduplication**
- Explain what a simple patent family is and why jurisdictional variants are redundant
- Explain the mechanism: the `families.simple_family.members` field is used; when one representative is processed, all other family member `lens_id`s are added to a skip set and dropped on encounter

**First independent claim extraction**
- Explain why first independent claims are included (encode the legal scope of the invention, not just its marketing summary)
- Describe the extraction logic: scans English claims list, picks the first claim whose text does NOT reference a prior claim number via regex (`\bclaim\s+\d+\b`); falls back to the first claim if all appear dependent

**Firm name normalization**
- Explain that applicant names are uppercased and trailing legal suffixes (INC, LLC, GmbH, S.p.A., etc.) are stripped via regex before constructing ownership edges

**Output**
- Three CSVs: `processed_patents.csv`, `citation_edges.csv`, `ownership_edges.csv`
- State what each contains

---

## 3.2 Semantic Representation — Patent Embeddings (`embed.py`)

**Model choice**
- Justify use of PatentSBERTa (`AI-Growth-Lab/PatentSBERTa`) over general-purpose SBERT; cite the domain-specific pretraining on patent corpora
- Max sequence length is 512 tokens (hard cap enforced in code)

**Input text construction**
- The input to the model is `title + " [SEP] " + first_claim` — or `title + " [SEP] " + abstract` when no first claim is available
- Explain why claims are preferred over abstracts (richer legal-technical content, broader coverage of inventive scope)

**Output**
- 768-dimensional dense vector per patent (PatentSBERTa output dimension)
- Embeddings are L2-normalized when running on GPU/MPS (`normalize_embeddings=True`)
- Saved as `patent_embeddings.npy` (shape: `[N_patents, 768]`)

**Hardware note** (optional, can go in Experiments section)
- Auto-detects CUDA / Apple MPS / CPU; uses multiprocessing pool on CPU

---

## 3.3 Heterogeneous Graph Construction (`build_graph.py`)

**Graph type**
- Explain why a heterogeneous graph is used: two distinct node types (patents, firms) with semantically different edge types
- Contrast with homogeneous citation-only graphs and why the ownership dimension adds value for thicket detection

**Node types and features**
- Patent nodes: feature matrix = `patent_embeddings.npy` (768-dim PatentSBERTa vectors)
- Firm nodes: placeholder scalar feature (value 0, 1-dim); note that firm identity is encoded structurally, not by content

**Edge types**
- `(patent, cites, patent)`: directed citation edges
- `(firm, owns, patent)`: directed ownership edges (applicant → patent)
- Both edge types are distinct relation types in PyG `HeteroData`

**Temporal filtering of citation edges**
- Explain the 0–5 year citation window: only citations where `citing_date − cited_date` is between 0 and 5 years are kept
- Justify: captures strategically relevant citations rather than retrospective or accidental cross-decade references
- State the volume reduction this produces (you must fill in numbers from your actual run)

**Implementation**
- Graph stored as PyTorch Geometric `HeteroData`, saved as `hetero_graph.pt`

---

## 3.4 Geometric Deep Learning Architecture (`model_dmon.py`)

**Model class: `DMON_DPR_Model`**
- A two-layer architecture: one GCNConv layer + one linear pooling layer
- GCNConv: 768 → 512 (patent node feature compression)
- Linear pool: 512 → K (outputs logits over K clusters per node)
- Soft cluster assignment matrix C is produced via row-wise softmax over pooling output

**What the model uses at training time**
- ONLY patent nodes and `(patent, cites, patent)` edges are fed to the GCN (`train.py` lines 13–14)
- Firm nodes and `(firm, owns, patent)` edges are NOT passed to the GCN encoder — they are used only in post-hoc analysis (`analyze.py`, `find_triples.py`)
- **DISCREPANCY**: Your draft section "GSL" implies the full heterogeneous graph is used in the GNN. The code does NOT do this. The GCN operates only on the patent–citation homogeneous subgraph. Clarify or correct.

**Soft vs. hard assignment**
- Training uses soft (probabilistic) assignments C ∈ ℝ^{N×K}
- At inference, hard assignments are derived via `argmax(C, dim=-1)`

---

## 3.5 Training Objective — DMoN-DPR Loss (`train.py`)

**Optimizer**: Adam, lr = 0.001, 200 epochs

**Five loss terms** (all need to be defined and motivated):

- `L_modularity`: rewards dense intra-cluster and sparse inter-cluster citation edges; formula uses the null model (degree-product baseline); normalized by `2 * n_edges`
- `L_collapse`: prevents degenerate solution where all patents are assigned to a single cluster; formula is `(√K / N) * ||sum_rows(C)||_F − 1`
- `L_distance` (weighted by `W_dist`): penalizes any pair of cluster centroids whose squared L2 distance in embedding space is less than epsilon (= 1.0 fixed); uses soft centroids `μ_k = C_k^T X / Σ_i C_ik`; formula uses `ReLU(ε − dist²)`; averaged over K(K−1) pairs
- `L_variance` (weighted by `W_var`): encourages dispersion in assignment probabilities across clusters; computed as negative mean of column-wise variance of C
- `L_entropy` (weight fixed at 0.1): pushes soft assignments toward one-hot (crisp); computed as negative mean of per-node assignment entropy

**Hyperparameters requiring tuning** (user-specified per run):
- K (number of clusters)
- W_dist (distance penalty weight)
- W_var (variance penalty weight)
- W_entropy is hardcoded at 0.1 — mention this is not tuned

**Run ID scheme**: `K{K}_Wd{W_dist}_Wv{W_var}` — enables reproducible multi-run comparison

---

## 3.6 Cluster Analysis & Thicket Identification (`analyze.py`)

**Hard assignment**: soft matrix C → argmax → integer cluster label per patent

**Per-cluster metrics computed**:

- **Cluster size** (n): number of patents assigned to cluster
- **Internal citations**: count of citation edges where both endpoints are in the same cluster
- **Clarkson density**: `internal_citations / (n*(n−1)/2)` — fraction of possible intra-cluster citation pairs that are realized
- **Unique firms**: count of distinct normalized firm names owning at least one patent in the cluster
- **Fragmentation index**: `unique_firms / cluster_size` — measures how many competing entities share overlapping IP

**Thicket candidate filter**:
- Size < 500 patents AND internal citations > 5
- Explain the rationale: avoids spuriously dense trivially-small clusters and ignores giant generic clusters

**Outputs**: `final_clusters_{run_id}.csv` (per-patent cluster label), `thicket_stats_{run_id}.csv` (per-cluster metrics)

---

## 3.7 Triple-Cycle Validation (`find_triples.py`)

**Purpose**: provide a structural, graph-theoretic validation of thicket candidates beyond density/fragmentation metrics

**What a triple is**:
- Three distinct firms A, B, C such that A cites B, B cites C, C cites A within the same cluster (directed cycle of length 3 in the inter-firm citation graph)
- Represents mutual blocking: each firm holds patents cited by, and citing, the others

**Construction**:
- Per cluster: build directed graph where nodes = firms and edges = inter-firm citation links (firm A → firm B if any patent owned by A cites any patent owned by B, within the cluster)
- Count all directed 3-cycles; deduplicate by sorting the firm triple

**Output metrics per cluster**:
- `num_triples`: count of distinct 3-cycles
- `firms_involved_in_triples`: number of firms participating in at least one cycle
- `total_firms_in_cluster`: total firms with ownership in the cluster

**Output**: `triple_stats_{run_id}.csv`

---

## Discrepancies Between Draft and Code

1. **Draft mentions "GSL" (Graph Structure Learning)** — the code performs no GSL. The graph topology is fixed (built from citation edges). There is no learned adjacency or topology inference. Either rename this section to "Graph Neural Network" or explain what you actually mean.

2. **Draft mentions PatentSBERTa with "CLAIMS"** but is vague — the code uses `title + [SEP] + first_independent_claim` (or abstract as fallback). Be explicit about this concatenation format and the independent-claim-first heuristic.

3. **Draft has no mention of the 5-loss training objective** — all five terms (modularity, collapse, distance, variance, entropy) must be described in the methodology; they are all in the code and are central to the method's novelty.

4. **Draft has no mention of temporal citation filtering** — the 0–5 year window in `build_graph.py` is a core design decision that must appear in the methodology.

5. **Draft has no mention of simple family deduplication** — this is implemented in `preprocess.py` and must be described; it directly affects dataset size and validity.

6. **Draft mentions "Clarkson density" and "fragmentation index" only vaguely ("uhh interpretare date")** — both metrics have exact formulas in `analyze.py` and must be formally defined.

7. **Draft has no mention of triple-cycle detection** (`find_triples.py`) — this is an entire validation step that must be covered in the methodology.

8. **Draft mentions using firm/applicant data ("patentees") but never explains the ownership edge construction** — must be described as a formal graph edge type.

9. **Draft mentions IPC/CPC codes as a possible data selection criterion** — the code has NO IPC/CPC filtering logic. If you used these codes in your Lens.org query, describe them here manually; they do not appear in any pipeline step.

10. **"W_entropy is hardcoded at 0.1"** — the draft says nothing about the entropy term. It is not user-tunable unlike K, W_dist, W_var. This asymmetry should be acknowledged.
