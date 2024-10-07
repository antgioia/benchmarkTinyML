import subprocess
import sys

def main():
    loader_path_Alexnet = sys.argv[1]
    loader_path_Resnet18 = sys.argv[2]
    loader_path_Vgg16 = sys.argv[3]
    
    try:
        print(f"\nInizio valutazione per il modello Alexnet...")
        subprocess.run(['python', 'techniques/quantization/alexnet/energyConsumed.py', loader_path_Alexnet], check=True)
        print(f"\nInizio valutazione per il modello Alexnet...")
        subprocess.run(['python', 'techniques/quantization/resnet18/energyConsumed.py', loader_path_Resnet18], check=True)
        print(f"\nInizio valutazione per il modello Alexnet...")
        subprocess.run(['python', 'techniques/quantization/vgg16/energyConsumed.py', loader_path_Vgg16], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore: {e}\n")

if __name__ == "__main__":
    main()
