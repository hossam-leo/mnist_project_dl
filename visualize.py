import matplotlib.pyplot as plt
import json
import os

def plot_history(history_files, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for file in history_files:
        with open(file, 'r') as f:
            history = json.load(f)
        name = os.path.basename(file).replace('_history.json', '')
        plt.plot(history['train_acc'], label=f'{name} Train Acc')
        plt.plot(history['val_acc'], label=f'{name} Val Acc', linestyle='--')
    
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for file in history_files:
        with open(file, 'r') as f:
            history = json.load(f)
        name = os.path.basename(file).replace('_history.json', '')
        plt.plot(history['train_loss'], label=f'{name} Train Loss')
        plt.plot(history['val_loss'], label=f'{name} Val Loss', linestyle='--')
    
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plots.png'))
    plt.show()

    # Individual plots for each model
    for file in history_files:
        with open(file, 'r') as f:
            history = json.load(f)
        name = os.path.basename(file).replace('_history.json', '')
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title(f'{name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'{name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_plots.png'))
        plt.close()

if __name__ == "__main__":
    history_files = ['results/Model_Adam_history.json', 'results/Model_SGD_history.json']
    plot_history(history_files)
    print("Visualizations generated in mnist_project/plots/")
