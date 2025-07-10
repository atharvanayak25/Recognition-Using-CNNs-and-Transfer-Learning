# Atharva Nayak - 002322653

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./fashion_experiment.csv')

metrics = ['accuracy', 'train_time']
for param in ['num_filters', 'dropout', 'batch_size', 'epochs']:
    grouped = df.groupby(param)[metrics].mean().reset_index()

    # Accuracy plot
    plt.figure()
    plt.plot(grouped[param], grouped['accuracy'], marker='o')
    plt.xlabel(param)
    plt.ylabel('Mean Test Accuracy')
    plt.title(f'Accuracy vs {param}')
    plt.savefig(f'accuracy_vs_{param}.png')
    plt.show()

    # Training time plot
    plt.figure()
    plt.plot(grouped[param], grouped['train_time'], marker='o')
    plt.xlabel(param)
    plt.ylabel('Mean Training Time (s)')
    plt.title(f'Training Time vs {param}')
    plt.savefig(f'time_vs_{param}.png')
    plt.show()

print('Saved all analysis plots.')
