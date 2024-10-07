import pandas as pd
import glob
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_csv_files_from_folder(folder_path):
    # Verifica se la directory esiste
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return pd.DataFrame()  # Ritorna un DataFrame vuoto

    # Trova tutti i file CSV nella cartella
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Verifica se sono stati trovati file CSV
    if not all_files:
        print(f"No CSV files found in: {folder_path}")
        return pd.DataFrame()  # Ritorna un DataFrame vuoto

    # Carica i file CSV in DataFrame
    dataframes = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Concatenate i DataFrame e restituisci
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()  # Ritorna un DataFrame vuoto se non ci sono dati

# Shapiro Wilk
def normality_test(data, method_name):
    stat, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk Test for {method_name}: Statistic={stat:.3f}, P-value={p_value:.3f}")
    return p_value

# Caricamento dei dati
low_rank_data = load_csv_files_from_folder('energyConsumed/lowRankApproximation/resnet18')
pre_data = load_csv_files_from_folder('energyConsumed/pre/resnet18')
quantization_data = load_csv_files_from_folder('energyConsumed/quantization/resnet18')
pruning_casuale_non_strutturato_data = load_csv_files_from_folder('energyConsumed/pruning/resnet18/pruned_casuale_non_strutturato')
pruning_global_data = load_csv_files_from_folder('energyConsumed/pruning/resnet18/pruned_global')
pruning_non_strutturato_data = load_csv_files_from_folder('energyConsumed/pruning/resnet18/pruned_non_strutturato')
pruning_strutturato_per_canali_data = load_csv_files_from_folder('energyConsumed/pruning/resnet18/pruned_strutturato_per_canali')

if 'energy_consumed' in low_rank_data.columns and 'energy_consumed' in pre_data.columns and 'energy_consumed' in quantization_data.columns:
    # Estrazione dell'enegia consumata
    low_rank_energy_consumed = low_rank_data['energy_consumed']
    pre_energy_consumed = pre_data['energy_consumed']
    quantization_energy_consumed = quantization_data['energy_consumed']
    pruning_casuale_non_strutturato_energy_consumed = pruning_casuale_non_strutturato_data['energy_consumed']
    pruning_global_energy_consumed = pruning_global_data['energy_consumed']
    pruning_non_strutturato_energy_consumed = pruning_non_strutturato_data['energy_consumed']
    pruning_strutturato_per_canali_energy_consumed = pruning_strutturato_per_canali_data['energy_consumed']

    p_value_low_rank = normality_test(low_rank_energy_consumed, 'Low-Rank Approximation')
    p_value_pre = normality_test(pre_energy_consumed, 'Pre')
    p_value_quantization = normality_test(quantization_energy_consumed, 'Quantization')
    p_value_pruning_cns = normality_test(pruning_casuale_non_strutturato_energy_consumed, 'Pruning (Casuale Non Strutturato)')
    p_value_pruning_g = normality_test(pruning_global_energy_consumed, 'Pruning (Globale)')
    p_value_pruning_ns = normality_test(pruning_non_strutturato_energy_consumed, 'Pruning (Non Strutturato)')
    p_value_pruning_spc = normality_test(pruning_strutturato_per_canali_energy_consumed, 'Pruning (Strutturato per Canali)')

    alpha = 0.05
    if p_value_low_rank <= alpha and p_value_pre <= alpha and p_value_quantization <= alpha and p_value_pruning_cns <= alpha and \
    p_value_pruning_g <= alpha and p_value_pruning_ns <= alpha and p_value_pruning_spc <= alpha:
        # Test Mann-Whitney
        t_stat_lr_pre, p_value_lr_pre = stats.mannwhitneyu(low_rank_energy_consumed, pre_energy_consumed)
        t_stat_quantization_pre, p_value_quantization_pre = stats.mannwhitneyu(quantization_energy_consumed, pre_energy_consumed)
        t_stat_pruning_pre, p_value_pruning_pre = stats.mannwhitneyu(pruning_casuale_non_strutturato_energy_consumed, pre_energy_consumed)
        t_stat_pruning_global_pre, p_value_pruning_global_pre = stats.mannwhitneyu(pruning_global_energy_consumed, pre_energy_consumed)
        t_stat_pruning_non_strutturato_pre, p_value_pruning_non_strutturato_pre = stats.mannwhitneyu(pruning_non_strutturato_energy_consumed, pre_energy_consumed)
        t_stat_pruning_strutturato_pre, p_value_pruning_strutturato_pre = stats.mannwhitneyu(pruning_strutturato_per_canali_energy_consumed, pre_energy_consumed)

        # Stampa dei risultati dei test
        print(f"T-statistic (Low Rank vs Pre): {t_stat_lr_pre:.3f}, P-value: {p_value_lr_pre:.3f}")
        print(f"T-statistic (Quantization vs Pre): {t_stat_quantization_pre:.3f}, P-value: {p_value_quantization_pre:.3f}")
        print(f"T-statistic (Pruning Casuale vs Pre): {t_stat_pruning_pre:.3f}, P-value: {p_value_pruning_pre:.3f}")
        print(f"T-statistic (Pruning Globale vs Pre): {t_stat_pruning_global_pre:.3f}, P-value: {p_value_pruning_global_pre:.3f}")
        print(f"T-statistic (Pruning Non Strutturato vs Pre): {t_stat_pruning_non_strutturato_pre:.3f}, P-value: {p_value_pruning_non_strutturato_pre:.3f}")
        print(f"T-statistic (Pruning Strutturato vs Pre): {t_stat_pruning_strutturato_pre:.3f}, P-value: {p_value_pruning_strutturato_pre:.3f}")

        # Preparazione dei dati per il boxplot
        boxplot_data1 = pd.DataFrame({
            'Method': ['TB'] * len(pre_energy_consumed) + \
                    ['TQ'] * len(quantization_energy_consumed) + \
                    ['TL'] * len(low_rank_energy_consumed),
            'energy_consumed': list(pre_energy_consumed) + \
                        list(quantization_energy_consumed) + \
                        list(low_rank_energy_consumed)
                        })
        
        boxplot_data2 = pd.DataFrame({
            'Method': ['TB'] * len(pre_energy_consumed) + \
                    ['TPCNS'] * len(pruning_casuale_non_strutturato_energy_consumed) + \
                    ['TPGNS'] * len(pruning_global_energy_consumed) + \
                    ['TPNS'] * len(pruning_non_strutturato_energy_consumed) + \
                    ['TPSPC'] * len(pruning_strutturato_per_canali_energy_consumed),
            'energy_consumed': list(pre_energy_consumed) + \
                        list(pruning_casuale_non_strutturato_energy_consumed) + \
                        list(pruning_global_energy_consumed) + \
                        list(pruning_non_strutturato_energy_consumed) + \
                        list(pruning_strutturato_per_canali_energy_consumed)
        })

        os.makedirs("energyConsumedAnalysis/result", exist_ok=True)
        # Visualizzazione con boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Method', y='energy_consumed', data=boxplot_data1)
        plt.ylabel('energy_consumed')
        plt.title("Comparison of resnet18 energy_consumed between Pre, Quantization and Low-Rank Approximation")
        plt.xticks(rotation=45)
        plt.savefig("energyConsumedAnalysis/result/pre_quantization_low_rank_approximation_resnet18.png")
        plt.show()
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Method', y='energy_consumed', data=boxplot_data2)
        plt.ylabel('energy_consumed')
        plt.title('Comparison of resnet18 energy_consumed between Pre and All Techniques of Pruning')
        plt.xticks(rotation=45)
        plt.savefig("energyConsumedAnalysis/result/pre_pruning_resnet18.png")
        plt.show()
else:
    print("La colonna 'energy_consumed' non è presente nei dati.")