import numpy as np
import pandas as pd

from results import results

architectures = ["distilhubert", "hubert", "wav2vec"]
datasets = ["local", "urbansound8k", "gender_voice", "instruments", "cats_dogs"]

table = pd.DataFrame(index=architectures, columns=datasets)

for arch in architectures:
    for dataset in datasets:
        acc_values = results[arch][dataset]
        
        mean_acc = np.mean(acc_values)
        std_acc = np.std(acc_values)
        
        table.loc[arch, dataset] = f"{mean_acc:.4f} ± {std_acc:.4f}"

print("\nTabla de resultados:\n")
print(table)

table.to_csv("results_table.csv")

table.to_excel("results_table.xlsx")