import json
import matplotlib.pyplot as plt

def plot_training_results(history_path="history.json"):
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {history_path}. Czy trening się zakończył?")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Trening')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Walidacja')
    plt.title('Strata (Loss) w czasie')
    plt.xlabel('Epoki')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Trening')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Walidacja')
    plt.title('Dokładność (Accuracy) w czasie')
    plt.xlabel('Epoki')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()