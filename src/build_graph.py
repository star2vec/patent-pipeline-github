import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch_geometric.data import HeteroData

def build_hetero_graph(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    # Încărcare dinamică din data_dir
    df_patents = pd.read_csv(f'{data_dir}/processed_patents.csv')
    df_ownership = pd.read_csv(f'{data_dir}/ownership_edges.csv')
    df_citations = pd.read_csv(f'{data_dir}/citation_edges.csv')
    patent_x = np.load(f'{data_dir}/patent_embeddings.npy')
    
    df_patents['date'] = pd.to_datetime(df_patents['date'])
    date_map = dict(zip(df_patents['patent_id'], df_patents['date']))
    
    def is_strategic_citation(row):
        citing_date = date_map.get(row['citing'])
        cited_date = date_map.get(row['cited'])
        if citing_date and cited_date:
            diff = (citing_date - cited_date).days / 365.25
            return 0 <= diff <= 5
        return False

    initial_count = len(df_citations)
    df_citations = df_citations[df_citations.apply(is_strategic_citation, axis=1)]
    print(f"Filtrare temporală: Am păstrat {len(df_citations)} din {initial_count} citații.")

    patent_id_map = {id: i for i, id in enumerate(df_patents['patent_id'])}
    unique_firms = df_ownership['firm'].unique()
    firm_id_map = {name: i for i, name in enumerate(unique_firms)}

    data = HeteroData()
    data['patent'].x = torch.from_numpy(patent_x).float()
    data['firm'].num_nodes = len(unique_firms)
    data['firm'].x = torch.zeros((len(unique_firms), 1))

    edge_index_cite = []
    for _, row in df_citations.iterrows():
        if row['citing'] in patent_id_map and row['cited'] in patent_id_map:
            edge_index_cite.append([patent_id_map[row['citing']], patent_id_map[row['cited']]])
    
    if edge_index_cite:
        data['patent', 'cites', 'patent'].edge_index = torch.tensor(edge_index_cite).t().contiguous()
    else:
        data['patent', 'cites', 'patent'].edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index_owns = []
    for _, row in df_ownership.iterrows():
        if row['firm'] in firm_id_map and row['patent_id'] in patent_id_map:
            edge_index_owns.append([firm_id_map[row['firm']], patent_id_map[row['patent_id']]])
    
    data['firm', 'owns', 'patent'].edge_index = torch.tensor(edge_index_owns).t().contiguous()

    # Salvare dinamică
    out_path = f'{data_dir}/hetero_graph.pt'
    torch.save(data, out_path)
    print(f"Succes! Graful a fost salvat în '{out_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HeteroGraph")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing CSVs and output PT")
    args = parser.parse_args()
    
    build_hetero_graph(args.data_dir)