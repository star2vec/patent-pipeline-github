import pandas as pd
import numpy as np
import torch
import argparse
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_csv, output_npy, batch_size=128):
    df = pd.read_csv(input_csv).fillna("")
    # Use first independent claim as primary text; fall back to abstract if unavailable
    claim_col = df['first_claim'].str.strip() if 'first_claim' in df.columns else pd.Series([""] * len(df))
    abstract_col = df['abstract'].str.strip()
    text_col = claim_col.where(claim_col != "", abstract_col)
    texts = (df['title'].fillna("") + " [SEP] " + text_col).tolist()
    
    # 1. OPTIMIZARE HARDWARE (Magia pentru Mac)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # Asta va activa GPU-ul de pe Apple Silicon (M1/M2/M3)
    else:
        device = "cpu"
        
    print(f"Se procesează {len(texts)} patente...")
    print(f"🚀 Se folosește acceleratorul hardware: {device.upper()}")

    # 2. Încărcare Model
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa', device=device)
    model.max_seq_length = 512
    
    # 3. Generare cu strategii diferite în funcție de hardware
    if device == "cpu":
        # Dacă suntem blocați pe CPU, folosim Multiprocessing (toate nucleele)
        print("Pornim procesarea multi-core pentru CPU. Așteaptă să pornească procesele...")
        pool = model.start_multi_process_pool()
        # encode_multi_process gestionează automat batch-urile și progresul
        embeddings = model.encode_multi_process(texts, pool, batch_size=batch_size)
        model.stop_multi_process_pool(pool)
        final_matrix = np.array(embeddings)
    else:
        # Dacă avem GPU/MPS, e mai rapid să lăsăm placa video să facă treaba secvențial
        print("Procesare rapidă pe placa video...")
        # Aici show_progress_bar e True pentru că `encode` simplu îl are integrat perfect

        final_matrix = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    
    # 4. Salvare matrice
    np.save(output_npy, final_matrix)
    print(f"Succes! Matricea de dimensiune {final_matrix.shape} a fost salvată în {output_npy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to processed_patents.csv")
    parser.add_argument('--out', type=str, required=True, help="Path to save patent_embeddings.npy")
    args = parser.parse_args()
    
    generate_embeddings(args.input, args.out)