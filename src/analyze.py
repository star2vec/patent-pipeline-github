import torch
import pandas as pd
import numpy as np
import os
import argparse
from model_dmon import DMON_DPR_Model

def run_analysis(data_dir, model_dir, K_val, W_dist_val, W_var_val, run_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graph_path = f'{data_dir}/hetero_graph.pt'
    if not os.path.exists(graph_path):
        print(f"Eroare: Graful nu există la {graph_path}")
        return
        
    data = torch.load(graph_path, weights_only=False)
    df_patents = pd.read_csv(f'{data_dir}/processed_patents.csv')
    df_ownership = pd.read_csv(f'{data_dir}/ownership_edges.csv')
    
    K = K_val
    X = data['patent'].x
    edge_index = data['patent', 'cites', 'patent'].edge_index
    
    model = DMON_DPR_Model(X.size(1), 512, K).to(device)
    
    model_path = f'{model_dir}/dmon_dpr_model_{run_id}.pth'
    if not os.path.exists(model_path):
        print(f"Eroare: Modelul nu a fost găsit la {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        _, c_matrix = model(X.to(device), edge_index.to(device))
        cluster_assignments = c_matrix.argmax(dim=-1).cpu().numpy()

    df_patents['cluster'] = cluster_assignments
    edge_index_np = edge_index.cpu().numpy()
    
    results = []
    
    for c in range(K):
        nodes_in_c = np.where(cluster_assignments == c)[0]
        n_c = len(nodes_in_c)
        
        if n_c < 2: continue 
        
        mask = np.isin(edge_index_np[0], nodes_in_c) & np.isin(edge_index_np[1], nodes_in_c)
        internal_citations = np.sum(mask)
        clarkson_density = internal_citations / (n_c * (n_c - 1) / 2) if n_c > 1 else 0
        
        cluster_patent_ids = df_patents.iloc[nodes_in_c]['patent_id']
        firme_unice = df_ownership[df_ownership['patent_id'].isin(cluster_patent_ids)]['firm'].nunique()
        frag_index = firme_unice / n_c if n_c > 0 else 0
        
        results.append({
            'cluster': c, 'size': n_c, 'internal_citations': internal_citations,
            'clarkson_density': round(clarkson_density, 5), 'unique_firms': firme_unice,
            'fragmentation_index': round(frag_index, 3)
        })

    df_stats = pd.DataFrame(results).sort_values(by='clarkson_density', ascending=False)
    
    print("\n--- Analiza DMON-DPR (High Fragmentation) ---")
    thickets = df_stats[(df_stats['size'] < 500) & (df_stats['internal_citations'] > 5)]
    print(thickets.head(15))
    
    df_stats['param_K'] = K_val
    df_stats['param_W_dist'] = W_dist_val
    df_stats['param_W_var'] = W_var_val

    df_patents.to_csv(f'{data_dir}/final_clusters_{run_id}.csv', index=False)
    df_stats.to_csv(f'{data_dir}/thicket_stats_{run_id}.csv', index=False)
    print(f"\nRezultate salvate în {data_dir}/thicket_stats_dmon.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--W_dist', type=float, required=True)
    parser.add_argument('--W_var', type=float, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    run_analysis(args.data_dir, args.model_dir, args.K, args.W_dist, args.W_var, args.run_id)