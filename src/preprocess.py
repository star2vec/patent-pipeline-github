import json
import re
import pandas as pd
import os
import argparse
from tqdm import tqdm

# Trailing legal entity suffixes to strip from firm names
_LEGAL = re.compile(
    r'[,\s]+(INC|LLC|LP|LLP|LTD|LIMITED|CORP(?:ORATION)?|CO(?:MPANY)?|GMBH|AG|SA|BV|NV|PLC|KK|KABUSHIKI\s+KAISHA'
    r'|S\.?P\.?A|S\.?R\.?L|AB|OY|AS|ASA|INDUSTRIES|HOLDINGS|GROUP|TECHNOLOGIES|IP|R&D)\.?$',
    re.IGNORECASE
)
_THE_PREFIX = re.compile(r'^THE\s+', re.IGNORECASE)
_ALIAS_PATH = os.path.join('raw-inputs', 'firm_aliases.csv')
_ALIASES = None  # lazily loaded list of (compiled_pattern, canonical)

def _load_aliases(path=_ALIAS_PATH):
    """Read raw-inputs/firm_aliases.csv into a list of (regex, canonical) tuples."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
    except Exception:
        return []
    out = []
    for _, row in df.iterrows():
        pat, canon = row.get('pattern', '').strip(), row.get('canonical', '').strip()
        if pat and canon:
            try:
                out.append((re.compile(pat, re.IGNORECASE), canon))
            except re.error:
                continue
    return out

def _normalize_firm(name):
    """Uppercase, strip suffixes/prefix iteratively, then apply alias map."""
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = _load_aliases()

    name = name.upper().strip()
    # Iteratively strip trailing legal suffixes so chains like "CO LTD" collapse fully
    prev = None
    while prev != name:
        prev = name
        name = _LEGAL.sub('', name).strip().rstrip(',.&').strip()
    name = _THE_PREFIX.sub('', name).strip()
    name = re.sub(r'\s+', ' ', name)

    # Apply first matching alias (manual subsidiary / abbreviation map)
    for pattern, canonical in _ALIASES:
        if pattern.match(name):
            return canonical
    return name

def _clean_claim(text):
    """Strip leading claim number ('1. ') and normalize whitespace."""
    text = re.sub(r'^\d+\.\s*', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def _extract_first_independent_claim(data):
    """Return the cleaned text of the first independent English claim, or '' if unavailable."""
    claims_list = data.get('claims', [])
    en_wrapper = next((c for c in claims_list if c.get('lang') == 'en'), None)
    if not en_wrapper:
        return ""
    individual_claims = en_wrapper.get('claims', [])
    # Independent claim: text does not reference a prior claim number
    for c in individual_claims:
        text = (c.get('claim_text') or [''])[0]
        if text and not re.search(r'\bclaim\s+\d+\b', text, re.IGNORECASE):
            return _clean_claim(text)
    # Fallback: all claims seem dependent — use the first one anyway
    if individual_claims:
        return _clean_claim((individual_claims[0].get('claim_text') or [''])[0])
    return ""

def preprocess_patents(input_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    patents, citations, ownership = [], [], []

    # Simple family deduplication: skip patents that are jurisdictional variants
    # of a family member we've already kept.
    skip_ids = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing Lens Data"):
            data = json.loads(line)
            patent_id = data.get('lens_id')

            # Skip jurisdictional duplicates identified from previously-processed families
            if patent_id in skip_ids:
                continue

            biblio = data.get('biblio', {})

            title_list = biblio.get('invention_title', [])
            title_text = next((t['text'] for t in title_list if t.get('lang') == 'en'), "")

            abstract_list = data.get('abstract', [])
            abstract_text = next((a['text'] for a in abstract_list if a.get('lang') == 'en'), "")

            first_claim = _extract_first_independent_claim(data)

            if not first_claim and not abstract_text:
                continue

            patents.append({
                'patent_id': patent_id, 'title': title_text,
                'abstract': abstract_text, 'first_claim': first_claim,
                'date': data.get('date_published')
            })

            # Mark all other simple-family members so they are skipped when encountered
            members = data.get('families', {}).get('simple_family', {}).get('members', [])
            for m in members:
                mlid = m.get('lens_id') or m.get('document_id', {}).get('lens_id')
                if mlid and mlid != patent_id:
                    skip_ids.add(mlid)

            applicants = biblio.get('parties', {}).get('applicants', [])
            for app in applicants:
                raw_name = app.get('extracted_name', {}).get('value', '')
                firm_name = _normalize_firm(raw_name) if raw_name else ''
                if len(firm_name) > 1:
                    ownership.append({'firm': firm_name, 'patent_id': patent_id})

            out_refs = biblio.get('references_cited', {}).get('citations', [])
            for ref in out_refs:
                cited_id = ref.get('lens_id') or ref.get('patcit', {}).get('lens_id')
                if cited_id:
                    citations.append({'citing': patent_id, 'cited': cited_id})

            in_refs = biblio.get('cited_by', {}).get('patents', [])
            for ref in in_refs:
                citing_id = ref.get('lens_id')
                if citing_id:
                    citations.append({'citing': citing_id, 'cited': patent_id})

    pd.DataFrame(patents).drop_duplicates('patent_id').to_csv(f'{out_dir}/processed_patents.csv', index=False)
    df_own = pd.DataFrame(ownership).drop_duplicates()
    df_own.to_csv(f'{out_dir}/ownership_edges.csv', index=False)
    pd.DataFrame(citations).drop_duplicates().to_csv(f'{out_dir}/citation_edges.csv', index=False)

    # Diagnostic: top firms by patent count (drives manual extension of firm_aliases.csv)
    if not df_own.empty:
        firm_counts = df_own['firm'].value_counts().head(100)
        firm_counts.to_csv(f'{out_dir}/firm_counts_top100.csv', header=['patent_count'])
        n_unique = df_own['firm'].nunique()
        print(f"Top 100 firms after normalization → {out_dir}/firm_counts_top100.csv "
              f"({n_unique} unique firms total). Eyeball the top rows; add patterns to "
              f"{_ALIAS_PATH} if you see obvious duplicates (e.g. SAMSUNG ELECTRONICS vs SAMSUNG DISPLAY).")

    print(f"Done! Kept {len(patents)} patents ({len(skip_ids)} family duplicates skipped) and {len(citations)} citation links.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Raw Patents")
    parser.add_argument('--input', type=str, required=True, help="Path to raw JSONL file")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save CSVs")
    args = parser.parse_args()

    preprocess_patents(args.input, args.out_dir)
