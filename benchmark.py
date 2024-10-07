import subprocess

try:
    subprocess.run(['python', 'preTechniques/benchmark.py'], check=True)
    subprocess.run(['python', 'techniques/quantization/benchmark.py'], check=True)
    subprocess.run(['python', 'techniques/pruning/benchmark.py'], check=True)
    subprocess.run(['python', 'techniques/lowRankApproximation/benchmark.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore nell'eseguire: {e}\n")