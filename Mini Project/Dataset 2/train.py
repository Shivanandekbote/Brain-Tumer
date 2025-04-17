
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model
import matplotlib.pyplot as plt

def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), dtype=tf.float32))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + 1e-15)
    recall = true_positives / (actual_positives + 1e-15)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return f1

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)



if __name__ == "__main__":
    ## Dataset
    path = "Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 50

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou, f1_score]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    history=model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)




    # After model.fit
    final_train_metrics = model.evaluate(train_dataset, steps=train_steps, verbose=0)
    final_valid_metrics = model.evaluate(valid_dataset, steps=valid_steps, verbose=0)

    report = f"Final Training Metrics:\nLoss: {final_train_metrics[0]:.4f}\nAccuracy: {final_train_metrics[1]:.4f}\nRecall: {final_train_metrics[2]:.4f}\nPrecision: {final_train_metrics[3]:.4f}\nIOU: {final_train_metrics[4]:.4f}\nF1-Score: {final_train_metrics[5]:.4f}\n\n"

    report += f"Final Validation Metrics:\nLoss: {final_valid_metrics[0]:.4f}\nAccuracy: {final_valid_metrics[1]:.4f}\nRecall: {final_valid_metrics[2]:.4f}\nPrecision: {final_valid_metrics[3]:.4f}\nIOU: {final_valid_metrics[4]:.4f}\nF1-Score: {final_valid_metrics[5]:.4f}"
    
    # Create 'files/report/' directory if it doesn't exist
    report_dir = 'files/report/'
    os.makedirs(report_dir, exist_ok=True)

# Save the report.txt in 'files/report/' directory
    with open(os.path.join(report_dir, "report.txt"), "w") as report_file:
        report_file.write(report)

                    # Plotting and saving individual plots
    plt.figure(figsize=(8, 6))

    # Accuracy plot
    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('files/plots/accuracy_plot.png')
    plt.show()

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('files/plots/loss_plot.png')
    plt.show()

    # Recall plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig('files/plots/recall_plot.png')
    plt.show()

    # Precision plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('files/plots/precision_plot.png')
    plt.show()

    # IOU plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['iou'], label='Train IOU')
    plt.plot(history.history['val_iou'], label='Validation IOU')
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig('files/plots/iou_plot.png')
    plt.show()

    # F1-Score plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['f1_score'], label='Train F1-Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1-Score')
    plt.title('Training and Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.savefig('files/plots/f1_score_plot.png')
    plt.show()
        

