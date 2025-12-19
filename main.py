import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Resize images from 32x32 to 75x75 (ResNet50 minimum input size)
x_train_resized = tf.image.resize(x_train, [75, 75])
x_test_resized = tf.image.resize(x_test, [75, 75])

# Convert to numpy arrays
x_train_resized = x_train_resized.numpy()
x_test_resized = x_test_resized.numpy()

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Load pretrained ResNet50 (without top classification layer)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(75, 75, 3)
)

# Freeze base model
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(75, 75, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(100, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with frozen base model...")
history1 = model.fit(
    x_train_resized, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Fine-tuning: unfreeze top layers
base_model.trainable = True
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nFine-tuning model...")
history2 = model.fit(
    x_train_resized, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test_resized, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Get predictions
y_pred = model.predict(x_test_resized)
y_pred_classes = np.argmax(y_pred, axis=1)

# Combine training histories
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(len(acc))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=9, color='r', linestyle='--', label='Fine-tuning Start')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=9, color='r', linestyle='--', label='Fine-tuning Start')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Save training history as CSV
history_df = pd.DataFrame({
    'epoch': list(epochs_range),
    'train_accuracy': acc,
    'val_accuracy': val_acc,
    'train_loss': loss,
    'val_loss': val_loss,
    'phase': ['frozen']*10 + ['fine-tuning']*10
})
history_df.to_csv('training_history.csv', index=False)

# Save summary results
results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'ResNet50',
    'dataset': 'CIFAR-100',
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'final_train_accuracy': float(acc[-1]),
    'final_val_accuracy': float(val_acc[-1]),
    'best_val_accuracy': float(max(val_acc)),
    'best_val_accuracy_epoch': int(np.argmax(val_acc)),
    'total_epochs': len(acc),
    'frozen_epochs': 10,
    'fine_tuning_epochs': 10,
    'input_shape': [75, 75, 3],
    'num_classes': 100,
    'batch_size': 64,
    'initial_lr': 0.001,
    'fine_tuning_lr': 0.0001
}

with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save detailed results
detailed_results = {
    'summary': results,
    'training_history': {
        'accuracy': [float(x) for x in acc],
        'val_accuracy': [float(x) for x in val_acc],
        'loss': [float(x) for x in loss],
        'val_loss': [float(x) for x in val_loss]
    }
}

with open('detailed_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=4)

# Save predictions
predictions_df = pd.DataFrame({
    'true_label': y_test.flatten(),
    'predicted_label': y_pred_classes,
    'correct': (y_test.flatten() == y_pred_classes).astype(int)
})
predictions_df.to_csv('predictions.csv', index=False)

# Save model
model.save('resnet50_cifar100.h5')

# Save model architecture summary
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("\n=== Results Saved ===")
print("- training_history.png")
print("- training_history.csv")
print("- results_summary.json")
print("- detailed_results.json")
print("- predictions.csv")
print("- resnet50_cifar100.h5")
print("- model_summary.txt")