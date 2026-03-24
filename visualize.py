import json
import matplotlib.pyplot as plt

def plot_training_results(history_path="history.json"):
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error file not found")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation')
    plt.title('Loss in time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation')
    plt.title('Accuracy in time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()