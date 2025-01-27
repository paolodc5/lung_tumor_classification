import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
from config import CONFIG
from data_loader_class import DataLoader
from logging_utils import app_logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import seaborn as sns
import os
import random

def global_library_setup():
    seed = CONFIG['general']['seed']

    tf.get_logger().setLevel('ERROR')

    gpus = tf.config.list_physical_devices('GPU')
    app_logger.info(f"Numero di GPU disponibili: {len(gpus)}")

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    app_logger.info(f"Seed globale impostato per la riproducibilità: {seed}")

def get_callbacks():
    """
    Metodo per addestrare il modello con i dati forniti da DataLoader.
    :param model: Modello Keras da addestrare.
    :param train_generator: DataLoader per il training.
    :param val_generator: DataLoader per la validazione.
    """

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]





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

    # Ottieni dati di test e etichette vere dal generatore
    test_data, test_labels = get_test_data_and_labels(test_generator)

    # Calcola accuracy e validation accuracy
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])

    # Previsioni del modello
    predictions = model.predict(test_data)
    predicted_classes = get_predicted_classes(predictions)

    # Metriche
    f1 = calculate_f1_score(test_labels, predicted_classes)
    test_acc = calculate_accuracy_score(test_labels, predicted_classes)
    conf_matrix = generate_confusion_matrix(test_labels, predicted_classes, class_names, output_dir)
    roc_auc = generate_roc_curve(test_labels, predictions, output_dir)

    # Salva il grafico dell'accuracy
    save_accuracy_plot(acc, val_acc, output_dir)

    # Dizionario di risultati finali
    results = {
        "accuracy": acc[-1] if acc else None,
        "val_accuracy": val_acc[-1] if val_acc else None,
        "f1_score": f1,
        "test_accuracy": test_acc,
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": roc_auc
    }
    app_logger.info(f"Risultati finali salvati correttamente in {output_dir}")

    return results



def get_test_data_and_labels(test_generator):
    """
    Estrae i dati e le label vere dal test generator.

    :param test_generator: Generatore di test.
    :return: Tupla (dati_test, label_test).
    """
    test_data = []
    test_labels = []

    for batch_data, batch_labels in test_generator:
        test_data.append(batch_data)
        test_labels.append(batch_labels)

    test_data = np.vstack(test_data)
    test_labels = np.vstack(test_labels)

    return test_data, test_labels


def get_predicted_classes(predictions):
    """
    Ottiene le classi previste come indice degli output con probabilità massima.

    :param predictions: Output delle previsioni del modello Keras.
    :return: Array di classi previste (indici).
    """
    return np.argmax(predictions, axis=1)


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
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Matrice di Confusione")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return conf_matrix


from sklearn.metrics import roc_curve, auc


def generate_roc_curve(test_labels, predictions, output_dir):
    """
    Genera e salva la curva ROC.

    :param test_labels: Label vere con one-hot encoding.
    :param predictions: Predizioni del modello (probabilità).
    :param output_dir: Directory di output.
    :return: AUC della curva ROC.
    """
    fpr, tpr, _ = roc_curve(test_labels.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    return roc_auc


def save_accuracy_plot(accuracy, val_accuracy, output_dir):
    """
    Genera e salva il grafico di accuracy e validation accuracy.

    :param accuracy: Lista delle accuracy durante il training.
    :param val_accuracy: Lista delle validation accuracy.
    :param output_dir: Directory di output.
    """
    epochs = range(1, len(accuracy) + 1)

    plt.figure()
    plt.plot(epochs, accuracy, 'b', label="Training Accuracy")
    plt.plot(epochs, val_accuracy, 'r', label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()







