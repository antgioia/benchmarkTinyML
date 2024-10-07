import subprocess

try:
    print("ANALISI ALEXNET")
    subprocess.run(['python', 'energyConsumedAnalysis/analysis/energyConsumedAnalysisAlexnet.py'], check=True)
    print("ANALISI RESNET18")
    subprocess.run(['python', 'energyConsumedAnalysis/analysis/energyConsumedAnalysisResnet18.py'], check=True)
    print("ANALISI VGG16")    
    subprocess.run(['python', 'energyConsumedAnalysis/analysis/energyConsumedAnalysisVgg16.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore nell'eseguire: {e}\n")