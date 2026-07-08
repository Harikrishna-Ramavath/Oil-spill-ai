"""
train.py

This script handles loading the synthetic/real dataset, preprocessing/augmenting
the images, training a classification model based on Transfer Learning with MobileNetV2,
evaluating the model, and generating visual plots (confusion matrix, ROC, curves).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def build_data_pipelines(config):
    """
    Loads dataset from directories and returns training and testing tf.data.Dataset objects.
    """
    dataset_dir = config["dataset_dir"]
    img_size = (config["img_height"], config["img_width"])
    batch_size = config["batch_size"]

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )

    # Configure data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])

    # Preprocessing wrapper
    # MobileNetV2 requires inputs in range [-1, 1], which tf.keras.applications.mobilenet_v2.preprocess_input handles.
    def preprocess_train(x, y):
        x = data_augmentation(x)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    def preprocess_test(x, y):
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    # Optimize datasets for performance
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds

def create_model(config):
    """
    Creates a Transfer Learning network using MobileNetV2 as the base.
    """
    img_size = (config["img_height"], config["img_width"], 3)
    
    # Load base model pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False

    # Build model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def lr_scheduler(epoch, lr):
    """Simple learning rate decay scheduler."""
    import math
    if epoch < 3:
        return float(lr)
    else:
        return float(lr * math.exp(-0.2))

def train_model():
    # Load configurations
    config = load_config()
    os.makedirs(config["models_dir"], exist_ok=True)

    # Build datasets
    train_ds, test_ds = build_data_pipelines(config)

    # Initialize model
    logger.info("Initializing MobileNetV2 model...")
    model = create_model(config)
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    model.summary()

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=config["model_path"],
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler_cb = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    # TensorBoard setup
    tb_log_dir = os.path.join(config["models_dir"], "logs")
    tensorboard_cb = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

    # Train
    logger.info("Starting model training...")
    history = model.fit(
        train_ds,
        epochs=config["epochs"],
        validation_data=test_ds,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb, tensorboard_cb]
    )

    # Save final evaluation metrics and plots
    logger.info("Evaluating model performance...")
    evaluate_and_plot(model, test_ds, history, config)

def evaluate_and_plot(model, test_ds, history, config):
    """
    Runs evaluations on test data and creates matplotlib visualization files.
    """
    models_dir = config["models_dir"]
    
    # Extract training metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    # Plot Accuracy Curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label='Training Accuracy', color='cyan', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='magenta', linewidth=2)
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label='Training Loss', color='cyan', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='magenta', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Run predictions on test dataset
    y_true = []
    y_pred_probs = []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().flatten())
        y_pred_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = (y_pred_probs >= config["classification_threshold"]).astype(int)

    # Compute stats
    cm = confusion_matrix(y_true, y_pred_labels)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate metrics
    accuracy = np.mean(y_true == y_pred_labels)
    tp = np.sum((y_true == 1) & (y_pred_labels == 1))
    fp = np.sum((y_true == 0) & (y_pred_labels == 1))
    fn = np.sum((y_true == 1) & (y_pred_labels == 0))
    tn = np.sum((y_true == 0) & (y_pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Save metrics JSON
    metrics_data = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]
    }
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Spill', 'Oil Spill'])
    plt.yticks(tick_marks, ['No Spill', 'Oil Spill'])
    
    # Label cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Plot ROC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "roc_curve.png"), dpi=150)
    plt.close()

    logger.info("Evaluation plots saved to models/ directory successfully!")

if __name__ == "__main__":
    train_model()
