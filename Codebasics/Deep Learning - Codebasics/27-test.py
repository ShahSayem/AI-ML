import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup and Configuration ---

# Define key parameters for the project
IMG_SIZE = (224, 224) # The input size MobileNetV2 was trained on
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# --- 2. Load and Prepare a Custom Dataset ---

# For this tutorial, we'll use the 'tf_flowers' dataset as our custom dataset.
# The `tfds.load` function automatically downloads and prepares the data.
# We split the data: 70% for training, 15% for validation, 15% for testing.
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    with_info=True,
    as_supervised=True,
)

# Get the number of classes from the dataset metadata
num_classes = metadata.features['label'].num_classes
class_names = metadata.features['label'].names
print(f"Dataset has {num_classes} classes: {class_names}")

# Create a function to resize and normalize the images
def format_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply the formatting function to each dataset
train_ds = train_ds.map(format_image).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(format_image).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(format_image).batch(BATCH_SIZE)

# Let's look at a few images from our dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.title(metadata.features['label'].names[np.argmax(label[0])] if len(label.shape) > 1 else metadata.features['label'].names[label[0]])
    plt.axis("off")
plt.suptitle("Sample Images from the Dataset")
plt.show()


# --- 3. The Core of Transfer Learning: Create the Base Model ---

# Load the MobileNetV2 model pre-trained on ImageNet
# Key Arguments for Transfer Learning:
# - include_top=False: This removes the final classification layer (the "head") of the model.
#   This is essential because the original head was trained to classify 1000 ImageNet classes,
#   but we need to classify our own flower classes.
# - weights='imagenet': This downloads the weights that were learned from the ImageNet dataset.
# - input_shape: This specifies the shape of our input images.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
# This is the most critical step in transfer learning. By setting trainable to False,
# we prevent the weights of the pre-trained layers from being updated during training.
# This preserves all the powerful features (like edge and texture detectors) that
# MobileNetV2 has already learned.
base_model.trainable = False

print("\nSummary of the frozen base model (MobileNetV2):")
base_model.summary()


# --- 4. Add a Custom Classification Head ---

# We will now stack our own classification layers on top of the frozen base model.
# Only these new layers will be trained on our flower dataset.
model = tf.keras.Sequential([
    base_model,  # The frozen, pre-trained feature extractor
    tf.keras.layers.GlobalAveragePooling2D(), # Pools the features into a single vector
    tf.keras.layers.Dropout(0.2),             # Dropout for regularization to prevent overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax') # Our new classification layer
])


# --- 5. Compile the Model ---

# The compile step configures the model for training.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy', # Use this loss for integer labels
    metrics=['accuracy']
)

print("\nSummary of the final model (MobileNetV2 + Custom Head):")
model.summary()


# --- 6. Train the Model ---

print("\nStarting training (only the new classification head will be trained)...")

# We train the model using the .fit() method.
# Keras will only update the weights of the trainable layers (our new Dense layer).
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

print("\nTraining finished.")


# --- 7. Evaluate and Visualize Results ---

print("\nEvaluating model on the unseen test set...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot the training history to see how the model learned
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()


# --- 8. Making a Prediction on a Single Image ---

print("\n--- Making a prediction on a new image ---")

# Download a sample image of a sunflower to test the model
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url)

# Load and preprocess the new image
# The steps must be exactly the same as for the training data
img = tf.keras.utils.load_img(sunflower_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.cast(img_array, tf.float32) / 255.0 # Normalize
img_batch = tf.expand_dims(img_array, 0) # Create a batch

# Get the model's prediction
predictions = model.predict(img_batch)
# The output 'predictions' is an array of probabilities for each class, e.g., [[0.05, 0.9, 0.03, 0.01, 0.01]]
print(f"Raw prediction output (probabilities): {predictions[0]}")

# Find the index of the class with the highest probability
predicted_class_index = np.argmax(predictions[0])
print(f"Predicted class index: {predicted_class_index}")

# Get the name of the class from our list of class names
predicted_class_name = class_names[predicted_class_index]
confidence = np.max(predictions[0])

# Display the result
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class_name} ({confidence:.2%})")
plt.axis("off")
plt.show()

