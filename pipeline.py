import os
import subprocess
import datetime
import sys

def run_research_pipeline():
    print("=== Patent Thicket Pipeline (DMON-DPR) ===")
    
    # 1. Datele de intrare de la utilizator
    tech_name = input("Introdu numele tehnologiei (ex: 5G_Telecom): ").strip().replace(" ", "_")
    input_jsonl = input("Introdu calea către fișierul .jsonl (ex: raw_inputs/patents.jsonl): ").strip()

    if not os.path.exists(input_jsonl):
        print(f"Eroare: Fișierul {input_jsonl} nu a putut fi găsit!")
        return

    # 2. Creare structură de foldere
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = f"results/{tech_name}_{timestamp}"
    data_dir = f"{base_dir}/data"
    model_dir = f"{base_dir}/models"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_file_path = f"{base_dir}/pipeline_log.txt"
    
    print(f"\n🚀 Pipeline inițializat! Rezultatele vor fi salvate în: {base_dir}")
    print(f"📝 Log-urile vor fi scrise în: {log_file_path}\n")

    # 3. Lista de pași (Adăugăm '-u' pentru a forța afișarea în timp real a terminalului)
    steps = [
        ("1. Preprocesare", ["python", "-u", "src/preprocess.py", "--input", input_jsonl, "--out_dir", data_dir]),
        ("2. Generare Embeddings", ["python", "-u", "src/embed.py", "--input", f"{data_dir}/processed_patents.csv", "--out", f"{data_dir}/patent_embeddings.npy"]),
        ("3. Construcție Graf", ["python", "-u", "src/build_graph.py", "--data_dir", data_dir]),
        ("4. Antrenare DMON-DPR", ["python", "-u", "src/train.py", "--data_dir", data_dir, "--model_dir", model_dir]),
        ("5. Analiză Clustere", ["python", "-u", "src/analyze.py", "--data_dir", data_dir, "--model_dir", model_dir]),
        ("6. Căutare Tripluri", ["python", "-u", "src/find_triples.py", "--data_dir", data_dir])
    ]

    # 4. Execuția pașilor cu Streaming Live
    with open(log_file_path, "w") as log_file:
        log_file.write(f"--- Pipeline start: {tech_name} la {datetime.datetime.now()} ---\n\n")

        for step_name, command in steps:
            time_now = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"\n[{time_now}] Executare: {step_name}...")
            log_file.write(f"--- {step_name} ---\n")
            log_file.write(f"Comandă: {' '.join(command)}\n")
            
            # Popen ne permite să citim output-ul linie cu linie în timp ce rulează
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in process.stdout:
                sys.stdout.write(line)  # Afișează în terminalul tău live
                log_file.write(line)    # Scrie în log-ul text
                sys.stdout.flush()
                
            process.wait()

            log_file.write("\n" + "="*40 + "\n\n")

            if process.returncode != 0:
                print(f"\n❌ Eroare critică la {step_name}. Procesul s-a oprit. Verifică {log_file_path} pentru detalii.")
                sys.exit(1)
            else:
                print(f"✅ {step_name} finalizat cu succes!")

    print(f"\n🎉 Pipeline complet! Verifică folderul {base_dir} pentru rezultate.")

if __name__ == "__main__":
    run_research_pipeline()