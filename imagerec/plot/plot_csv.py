import pandas as pd
import matplotlib.pyplot as plt
import datetime
import logging
logger = logging.getLogger(__name__)

def plot_loss_accuracy(csv_path):
    """
    Plot training and validation loss and accuracy from a CSV file.

    Args:
        csv_path (str): Path to the CSV log file containing:
            - "Epoch"
            - "Train Loss"
            - "Validation Loss"
            - "Train Accuracy"
            - "Validation Accuracy"
    """
    logger.info(f"Loading CSV data for loss/accuracy plot: {csv_path}")
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
    ax1.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["Epoch"], df["Train Accuracy"], label="Train Accuracy")
    ax2.plot(df["Epoch"], df["Validation Accuracy"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy %")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_learning_rate(csv_path):
    """
    Plot the learning rate over training epochs.

    Args:
        csv_path (str): Path to the CSV log file containing:
            - "Epoch"
            - "Learning Rate"
    """
    logger.info(f"Loading CSV data for learning rate plot: {csv_path}")
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["Learning Rate"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_duration(csv_path):
    """
    Plot epoch durations in seconds over training epochs.

    Args:
        csv_path (str): Path to the CSV log file containing:
            - "Epoch"
            - "Duration"
    """
    logger.info(f"Loading CSV data for duration plot: {csv_path}")
    df = pd.read_csv(csv_path)

    durations_in_seconds = df["Duration"].apply(
        lambda x: datetime.timedelta(
            hours=int(x.split(":")[0]),
            minutes=int(x.split(":")[1]),
            seconds=float(x.split(":")[2])
        ).total_seconds()
    )

    plt.figure(figsize=(12, 8))
    plt.plot(df["Epoch"], durations_in_seconds, label="Epoch Duration (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Duration (seconds)")
    plt.title("Epoch Duration over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()