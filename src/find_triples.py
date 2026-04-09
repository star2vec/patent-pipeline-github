import pandas as pd
import networkx as nx
import argparse
from itertools import combinations

def find_thicket_triples(data_dir, run_id):
    # Schimbă din: df_clusters = pd.read_csv(f'{data_dir}/final_clusters_dmon.csv')
    df_clusters = pd.read_csv(f'{data_dir}/final_clusters_{run_id}.csv')
    df_citations = pd.read_csv(f'{data_dir}/citation_edges.csv')
    df_ownership = pd.read_csv(f'{data_dir}/ownership_edges.csv')

    patent_to_firm = dict(zip(df_ownership['patent_id'], df_ownership['firm']))
    
    results = []

    for cluster_id in df_clusters['cluster'].unique():
        cluster_patents = df_clusters[df_clusters['cluster'] == cluster_id]['patent_id'].tolist()
        
        G_firms = nx.DiGraph()
        
        relevant_cites = df_citations[
            df_citations['citing'].isin(cluster_patents) & 
            df_citations['cited'].isin(cluster_patents)
        ]

        for _, row in relevant_cites.iterrows():
            firm_a = patent_to_firm.get(row['citing'])
            firm_b = patent_to_firm.get(row['cited'])
            
            if firm_a and firm_b and firm_a != firm_b:
                G_firms.add_edge(firm_a, firm_b)

        triples = []
        for a in G_firms.nodes():
            for b in G_firms.successors(a):
                for c in G_firms.successors(b):
                    if G_firms.has_edge(c, a):
                        triple = tuple(sorted([a, b, c]))
                        if triple not in triples:
                            triples.append(triple)

        if len(triples) > 0:
            firms_in_triples = set([firm for t in triples for firm in t])
            results.append({
                'cluster': cluster_id,
                'num_triples': len(triples),
                'firms_involved_in_triples': len(firms_in_triples),
                'total_firms_in_cluster': len(set([patent_to_firm.get(p) for p in cluster_patents if patent_to_firm.get(p)]))
            })

    if results:
        df_triples = pd.DataFrame(results).sort_values(by='num_triples', ascending=False)
        print("\n--- Validare prin Metoda Triplurilor ---")
        print(df_triples.head(10))
        df_triples.to_csv(f'{data_dir}/triple_stats_{run_id}.csv', index=False)
        print(f"Salvat în {data_dir}/triple_stats.csv")
    else:
        print("\nNu a fost găsit niciun triplu în clusterele analizate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    find_thicket_triples(args.data_dir, args.run_id)