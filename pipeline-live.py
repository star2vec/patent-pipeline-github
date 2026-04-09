import os
import subprocess
import sys

def run_research_pipeline():
    print("=== Patent Thicket Pipeline LIVE (Smart Cache & Versioning) ===")
    
    # 1. Numele proiectului (folderul fix)
    tech_name = input("Introdu numele tehnologiei (ex: semicon-1998-2012): ").strip().replace(" ", "_")
    base_dir = f"results/{tech_name}"
    data_dir = f"{base_dir}/data"
    model_dir = f"{base_dir}/models"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Verificare Smart Cache (Avem deja embedding-urile?)
    embeddings_path = f"{data_dir}/patent_embeddings.npy"
    skip_embeddings = os.path.exists(embeddings_path)

    if skip_embeddings:
        print(f"\n✅ S-au găsit date preprocesate în {data_dir}!")
        print("⏭️  Se sare peste Etapele 1, 2 și 3 (Timp salvat: ~95%).")
    else:
        print(f"\n⚠️ Nu s-au găsit date pentru {tech_name}. Vom rula de la zero.")
        input_jsonl = input("Introdu calea către fișierul .jsonl (ex: raw_inputs/semicon.jsonl): ").strip()
        if not os.path.exists(input_jsonl):
            print(f"Eroare: Fișierul {input_jsonl} nu există!")
            return

    # 3. Setare Hiper-Parametri (Aici controlezi modelul)
    print("\n--- Parametrii Modelului DMON-DPR ---")
    val_K = input("K (număr clustere, ex: 100 pt teste macro, 300 pt micro): ").strip() or "100"
    val_W_dist = input("W_dist (greutatea distanței, ex: 1.0 sau 10.0): ").strip() or "1.0"
    val_W_var = input("W_var (dispersia asignărilor, ex: 1.0): ").strip() or "1.0"
    
    # Creăm un ID unic pentru această rulare
    run_id = f"K{val_K}_Wd{val_W_dist}_Wv{val_W_var}"
    log_file_path = f"{base_dir}/pipeline_log_{run_id}.txt"
    
    print(f"\n🚀 Începem rularea versiunii: [{run_id}]")

    # 4. Asamblarea Etapelelor
    steps = []
    
    if not skip_embeddings:
        steps.extend([
            ("1. Preprocesare", ["python", "-u", "src/preprocess.py", "--input", input_jsonl, "--out_dir", data_dir]),
            ("2. Generare Embeddings", ["python", "-u", "src/embed.py", "--input", f"{data_dir}/processed_patents.csv", "--out", embeddings_path]),
            ("3. Construcție Graf", ["python", "-u", "src/build_graph.py", "--data_dir", data_dir])
        ])
    
    # Etapele care rulează de fiecare dată, folosind noii parametri
    steps.extend([
        ("4. Antrenare DMON-DPR", ["python", "-u", "src/train.py", "--data_dir", data_dir, "--model_dir", model_dir, "--K", val_K, "--W_dist", val_W_dist, "--W_var", val_W_var, "--run_id", run_id]),
        ("5. Analiză Clustere", ["python", "-u", "src/analyze.py", "--data_dir", data_dir, "--model_dir", model_dir, "--K", val_K, "--W_dist", val_W_dist, "--W_var", val_W_var, "--run_id", run_id]),
        ("6. Căutare Tripluri", ["python", "-u", "src/find_triples.py", "--data_dir", data_dir, "--run_id", run_id])
    ])

    # 5. Execuție Live
    with open(log_file_path, "w") as log_file:
        for step_name, command in steps:
            print(f"\n[{run_id}] Executare: {step_name}...")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
                sys.stdout.flush()
            process.wait()

            if process.returncode != 0:
                print(f"\n❌ Eroare critică la {step_name}. Proces oprit.")
                sys.exit(1)

    print(f"\n🎉 Versiunea {run_id} finalizată! Găsești CSV-urile în {data_dir}")

if __name__ == "__main__":
    run_research_pipeline()