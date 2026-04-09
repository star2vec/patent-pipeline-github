from flask import Flask, request, Response
import os
import subprocess
import sys

app = Flask(__name__)

# 1. The Frontend Webpage (HTML + JavaScript)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Patent Thicket Pipeline</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { background-color: #28a745; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background-color: #218838; }
        #terminal { background-color: #1e1e1e; color: #00ff00; padding: 15px; margin-top: 20px; border-radius: 4px; height: 400px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🚀 Patent Thicket Pipeline (DMON-DPR)</h2>
        
        <label>Numele tehnologiei:</label>
        <input type="text" id="tech_name" value="semicon-1998-2012">
        
        <label>Numele fișierului .jsonl (din folderul raw_inputs):</label>
        <input type="text" id="input_jsonl" value="raw_inputs/semicon.jsonl">
        
        <label>K (număr clustere):</label>
        <input type="text" id="val_K" value="100">
        
        <label>W_dist (greutatea distanței):</label>
        <input type="text" id="val_W_dist" value="1.0">
        
        <label>W_var (dispersia asignărilor):</label>
        <input type="text" id="val_W_var" value="1.0">
        
        <button onclick="startPipeline()">Start Processing</button>
        
        <div id="terminal">Așteptare comenzi...</div>
    </div>

    <script>
        function startPipeline() {
            const terminal = document.getElementById('terminal');
            terminal.innerHTML = "Inițializare pipeline...\\n";
            
            const params = new URLSearchParams({
                tech_name: document.getElementById('tech_name').value,
                input_jsonl: document.getElementById('input_jsonl').value,
                val_K: document.getElementById('val_K').value,
                val_W_dist: document.getElementById('val_W_dist').value,
                val_W_var: document.getElementById('val_W_var').value
            });

            // Conectare la fluxul de date live (Server-Sent Events)
            const eventSource = new EventSource('/run_pipeline?' + params.toString());
            
            eventSource.onmessage = function(event) {
                terminal.innerHTML += event.data + "\\n";
                terminal.scrollTop = terminal.scrollHeight; // Auto-scroll
                
                if (event.data.includes("🎉 Versiunea") || event.data.includes("❌ Eroare")) {
                    eventSource.close();
                }
            };
        }
    </script>
</body>
</html>
"""

# 2. The API Endpoints
@app.route('/')
def home():
    return HTML_PAGE

@app.route('/run_pipeline')
def run_pipeline():
    # Preluăm variabilele din formularul web
    tech_name = request.args.get('tech_name', 'default_tech').strip().replace(" ", "_")
    input_jsonl = request.args.get('input_jsonl', '').strip()
    val_K = request.args.get('val_K', '100').strip()
    val_W_dist = request.args.get('val_W_dist', '1.0').strip()
    val_W_var = request.args.get('val_W_var', '1.0').strip()

    # Generator pentru a trimite datele bucată cu bucată către browser
    def generate_output():
        yield f"data: === Patent Thicket Pipeline LIVE (Smart Cache & Versioning) ===\n\n"
        
        base_dir = f"results/{tech_name}"
        data_dir = f"{base_dir}/data"
        model_dir = f"{base_dir}/models"
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        embeddings_path = f"{data_dir}/patent_embeddings.npy"
        skip_embeddings = os.path.exists(embeddings_path)

        if skip_embeddings:
            yield f"data: ✅ S-au găsit date preprocesate în {data_dir}!\n\n"
            yield f"data: ⏭️ Se sare peste Etapele 1, 2 și 3 (Timp salvat: ~95%).\n\n"
        else:
            yield f"data: ⚠️ Nu s-au găsit date pentru {tech_name}. Vom rula de la zero.\n\n"
            if not os.path.exists(input_jsonl):
                yield f"data: ❌ Eroare: Fișierul {input_jsonl} nu există pe server!\n\n"
                return

        run_id = f"K{val_K}_Wd{val_W_dist}_Wv{val_W_var}"
        yield f"data: 🚀 Începem rularea versiunii: [{run_id}]\n\n"

        steps = []
        if not skip_embeddings:
            steps.extend([
                ("1. Preprocesare", ["python", "-u", "src/preprocess.py", "--input", input_jsonl, "--out_dir", data_dir]),
                ("2. Generare Embeddings", ["python", "-u", "src/embed.py", "--input", f"{data_dir}/processed_patents.csv", "--out", embeddings_path]),
                ("3. Construcție Graf", ["python", "-u", "src/build_graph.py", "--data_dir", data_dir])
            ])
        
        steps.extend([
            ("4. Antrenare DMON-DPR", ["python", "-u", "src/train.py", "--data_dir", data_dir, "--model_dir", model_dir, "--K", val_K, "--W_dist", val_W_dist, "--W_var", val_W_var, "--run_id", run_id]),
            ("5. Analiză Clustere", ["python", "-u", "src/analyze.py", "--data_dir", data_dir, "--model_dir", model_dir, "--K", val_K, "--W_dist", val_W_dist, "--W_var", val_W_var, "--run_id", run_id]),
            ("6. Căutare Tripluri", ["python", "-u", "src/find_triples.py", "--data_dir", data_dir, "--run_id", run_id])
        ])

        # Execuția pașilor și trimiterea textului către browser (yield)
        for step_name, command in steps:
            yield f"data: \n\n"
            yield f"data: [{run_id}] Executare: {step_name}...\n\n"
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                # Formatăm linia pentru a fi citită corect de browser (format Server-Sent Events)
                clean_line = line.strip().replace('\n', ' ')
                yield f"data: {clean_line}\n\n"
            
            process.wait()

            if process.returncode != 0:
                yield f"data: ❌ Eroare critică la {step_name}. Proces oprit.\n\n"
                return

        yield f"data: 🎉 Versiunea {run_id} finalizată! Găsești rezultatele în {data_dir}\n\n"

    # Returnăm generatorul ca pe un stream continuu
    return Response(generate_output(), mimetype='text/event-stream')

if __name__ == "__main__":
    # Pornim serverul web pe portul 5000
    app.run(host='0.0.0.0', port=5000, debug=True)