import subprocess

def esegui_script(script_name):
    try:
        # Esegui lo script Python
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        # Stampa l'output dello script
        print(f"Output di {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        # Gestisci errori durante l'esecuzione
        print(f"Errore durante l'esecuzione di {script_name}:\n{e.stderr}")

# Esegui i tre script
esegui_script('techniques/lowRankApproximation/alexnet/lowRankApproximationAlexNet.py')
esegui_script('techniques/lowRankApproximation/resnet18/lowRankApproximationResNet18.py')
esegui_script('techniques/lowRankApproximation/vgg16/lowRankApproximationVgg16.py')