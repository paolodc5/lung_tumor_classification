
import numpy as np
import random
from config import CONFIG
from data_loader_class import DataLoader
from global_utils import convert_dict_to_json
from logging_utils import app_logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
import seaborn as sns
import os


def evaluate_results(history, model, class_names=None):
    """
    Funzione principale per valutare le performance del modello.
    Richiama funzioni modulari per calcolare accuracy, validation accuracy,
    F1 score, matrice di confusione e curva ROC.

    :param history: Oggetto Keras History, contenente le metriche durante il training.
    :param model: Modello Keras addestrato.
    :param class_names: Lista di nomi delle classi (facoltativo).
    :return: Dictionary con tutte le metriche richieste.
    """
    app_logger.info("Inizio valutazione del modello...")
    test_generator = DataLoader(split='test')

    if not history or not hasattr(history, 'history'):
        raise TypeError("History must be un oggetto valido di tipo Keras History")

    # Percorso output per i grafici
    output_dir = CONFIG['output']['save_path']
    os.makedirs(output_dir, exist_ok=True)

    # salva il json dict history
    convert_dict_to_json(history.history, file_name='history.json')

    # Ottieni dati di test e etichette vere dal generatore
    test_data, test_labels = get_test_data_and_labels(test_generator)

    # Calcola accuracy e validation accuracy
    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    train_f1 = history.history.get('f1_score', [])
    val_f1 = history.history.get('val_f1_score', [])
    train_precision = history.history.get('precision', [])
    val_precision = history.history.get('val_precision', [])
    train_recall = history.history.get('recall', [])
    val_recall = history.history.get('val_recall', [])

    # Previsioni del modello
    predictions = model.predict(test_data)
    predicted_classes = get_predicted_classes(predictions)

    # Metriche
    test_f1 = calculate_f1_score(test_labels, predicted_classes)
    test_acc = calculate_accuracy_score(test_labels, predicted_classes)
    test_precision = calculate_precision_score(test_labels, predicted_classes)
    test_recall = calculate_recall_score(test_labels, predicted_classes)
    conf_matrix = generate_confusion_matrix(test_labels, predicted_classes, class_names, output_dir)
    test_roc_auc = generate_roc_curve(test_labels, predictions, output_dir)
    print(f"roc auc: {test_roc_auc}")

    # Salva il grafico dell'accuracy
    save_accuracy_plot(train_acc, val_acc, output_dir)

    # Dizionario di risultati finali
    results = {
        "final_accuracy": train_acc[-1] if train_acc else None,
        "final_val_accuracy": val_acc[-1] if val_acc else None,
        "final_f1_score": train_f1[-1] if train_f1 else None,
        "final_val_f1_score": val_f1[-1] if val_f1 else None,
        "final_precision": train_precision[-1] if train_precision else None,
        "final_val_precision": val_precision[-1] if val_precision else None,
        "final_recall": train_recall[-1] if train_recall else None,
        "final_val_recall": val_recall[-1] if val_recall else None,
        "test_accuracy": test_acc,
        "test_f1_score": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": test_roc_auc
    }
    convert_dict_to_json(results) # This saves also the file
    app_logger.info(f"Risultati finali salvati correttamente in {output_dir}")

    return results



def get_test_data_and_labels(test_generator):
    """
    Estrae i dati e le etichette vere dal test generator.

    :param test_generator: Generatore di batch per il test set.
    :return: Tupla contenente tutti i dati di test e tutte le label (test_data, test_labels).
    """
    test_data = []
    test_labels = []

    for batch_data, batch_labels in test_generator:
        # Garantire che sia `batch_data` che `batch_labels` siano array di NumPy
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        # Aggiungere i dati e le label alla lista principale
        test_data.append(batch_data)
        test_labels.append(batch_labels)

    # Concatenare i batch invece di impilarli
    try:
        test_data = np.concatenate(test_data, axis=0)  # Concatenazione lungo il batch axis
        test_labels = np.concatenate(test_labels, axis=0)
    except ValueError as e:
        error = ValueError(f"Errore durante la concatenazione: {e}")
        app_logger.error(str(error))
        raise

    return test_data, test_labels


def get_predicted_classes(predictions, threshold=0.5):
    """
    Ottiene le classi previste per un task di classificazione binaria.

    :param predictions: Output delle previsioni del modello Keras (valori di probabilità tra 0 e 1).
    :param threshold: Soglia per classificare le probabilità (default: 0.5).
    :return: Array di classi previste (0 o 1).
    """
    return (predictions >= threshold).astype(int)


def calculate_f1_score(true_classes, predicted_classes):
    """
    Calcola il F1 score ponderato.

    :param true_classes: Classi vere (etichette).
    :param predicted_classes: Classi previste dal modello.
    :return: F1 score ponderato.
    """
    return f1_score(true_classes, predicted_classes, average='weighted')

def calculate_accuracy_score(true_classes, predicted_classes):
    """
    Calcola l'accuracy sul test set.

    :param true_classes: Array delle classi vere (etichette reali).
    :param predicted_classes: Array delle classi previste dal modello.
    :return: Accuracy (float).
    """
    if len(true_classes) != len(predicted_classes):
        raise ValueError("La dimensione di true_classes e predicted_classes deve essere uguale.")

    correct_predictions = sum(true == pred for true, pred in zip(true_classes, predicted_classes))
    accuracy = correct_predictions / len(true_classes)

    return accuracy


def calculate_precision_score(true_labels, predicted_labels):
    """
    Calcola la precision dei risultati del modello.

    :param true_labels: Array delle etichette reali.
    :param predicted_labels: Array delle etichette previste dal modello.
    :return: Precision score come valore float.
    """
    return precision_score(true_labels, predicted_labels, average='weighted')


def calculate_recall_score(true_labels, predicted_labels):
    """
    Calcola il recall dei risultati del modello.

    :param true_labels: Array delle etichette reali.
    :param predicted_labels: Array delle etichette previste dal modello.
    :return: Recall score come valore float.
    """
    return recall_score(true_labels, predicted_labels, average='weighted')

def generate_confusion_matrix(true_classes, predicted_classes, class_names, output_dir):
    """
    Genera la matrice di confusione e salva il grafico.

    :param true_classes: Classi vere.
    :param predicted_classes: Classi previste.
    :param class_names: Nomi delle classi (facoltativo).
    :param output_dir: Directory di output.
    :return: Matrice di confusione come array.
    """
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
    )
    plt.title("Matrice di Confusione")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return conf_matrix


def generate_roc_curve(test_labels, predictions, output_path):
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fpr, tpr, _ = roc_curve(test_labels.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, linestyle='-', linewidth=2.5, label=f"AUC = {roc_auc:.2f}", color="#1f77b4")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color="grey")

    plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight='bold', pad=15)

    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.savefig(os.path.join(output_path, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc


def save_accuracy_plot(accuracy, val_accuracy, output_dir):
    """
    Genera e salva un grafico di training e validation accuracy con uno stile migliorato.

    :param accuracy: Lista delle accuracy durante il training.
    :param val_accuracy: Lista delle validation accuracy.
    :param output_dir: Directory di output.
    """
    sns.set_style("whitegrid")  # Stile pulito
    sns.set_context("talk")  # Testo ben leggibile

    epochs = np.arange(1, len(accuracy) + 1)

    plt.figure(figsize=(8, 5))  # Dimensioni ottimali
    plt.plot(epochs, accuracy, marker='o', linestyle='-', linewidth=2, markersize=6, label="Training Accuracy",
             color="#1f77b4")
    plt.plot(epochs, val_accuracy, marker='s', linestyle='--', linewidth=2, markersize=6, label="Validation Accuracy",
             color="#ff7f0e")

    plt.xticks(np.arange(1, len(accuracy) + 1, step=5))  # Mostra solo numeri interi sulle epoche
    plt.xlabel("Epochs", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title("Training & Validation Accuracy", fontsize=16, fontweight='bold', pad=15)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)  # Griglia sottile

    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()





if __name__ == '__main__':
    test_generator = iter(DataLoader(split='test'))
    test_data, test_labels = get_test_data_and_labels(test_generator)
    print(test_data.shape)
    print(test_labels.shape)











