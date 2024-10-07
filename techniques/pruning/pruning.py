import os
import subprocess

# Definisci le directory e i file per ogni modello e tecnica di pruning
models = ['alexnet', "resnet18", "vgg16"]
techniques = {
    'pruningCasualeNonStrutturato': 'pruningCasualeNonStrutturato{}.py',
    'pruningGlobaleNonStrutturato': 'pruningGlobaleNonStrutturato{}.py',
    'pruningNonStrutturato': 'pruningNonStrutturato{}.py',
    'pruningStrutturatoPerCanali': 'pruningStrutturatoPerCanali{}.py'
}

# Base path della directory (puoi modificarlo se i tuoi script si trovano in un'altra directory)
base_path = './techniques/pruning'

def esegui_script(script_path):
    """Esegue uno script Python specificato dal percorso."""
    try:
        # Esegui il comando Python per eseguire lo script
        print(f"Eseguendo {script_path}...")
        subprocess.run(['python', script_path], check=True)
        print(f"Completato: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'eseguire {script_path}: {e}\n")

def esegui_pruning_per_modello(modello):
    """Esegue tutti i file di pruning per un modello specifico."""
    for tecnica, file_template in techniques.items():
        # Crea il percorso completo del file da eseguire
        file_name = file_template.format(modello.capitalize())
        script_path = os.path.join(base_path, modello, file_name)
        
        # Verifica che il file esista prima di eseguirlo
        if os.path.exists(script_path):
            esegui_script(script_path)
        else:
            print(f"File non trovato: {script_path}\n")

def main():
    """Funzione principale per eseguire il pruning per tutti i modelli."""
    for modello in models:
        print(f"\nInizio pruning per il modello {modello}...")
        esegui_pruning_per_modello(modello)
        print(f"Completato pruning per il modello {modello}\n")

if __name__ == "__main__":
    main()
