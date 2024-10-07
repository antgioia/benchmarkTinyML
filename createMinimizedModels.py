import subprocess

try:
    subprocess.run(['python', 'techniques/quantization/quantization.py'], check=True)
    subprocess.run(['python', 'techniques/pruning/pruning.py'], check=True)
    subprocess.run(['python', 'techniques/lowRankApproximation/lowRankApproximation.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore nell'eseguire: {e}\n")