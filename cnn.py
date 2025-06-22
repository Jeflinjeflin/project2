import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog

# Set environment variable to avoid oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define your fruit classes based on your dataset
FRUIT_CLASSES = ['apple', 'banana', 'grapes', 'kiwi', 'mango', 'strawberry']

def create_cnn_model(num_classes=6, input_shape=(224, 224, 3)):
    """
    Create CNN model for fruit classification
    """
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten and Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data(data_dir, img_size=(224, 224), test_size=0.2):
    """
    Load images from fruit_dataset folder and preprocess them
    """
    images = []
    labels = []
    class_names = []
    
    print("Loading dataset...")
    
    # Get all fruit folders
    for class_idx, fruit_folder in enumerate(sorted(os.listdir(data_dir))):
        fruit_path = os.path.join(data_dir, fruit_folder)
        
        if os.path.isdir(fruit_path):
            class_names.append(fruit_folder)
            print(f"Loading {fruit_folder} images...")
            
            # Load all images from this fruit folder
            for img_file in os.listdir(fruit_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img_path = os.path.join(fruit_path, img_file)
                    
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize(img_size)
                        img_array = np.array(img) / 255.0  # Normalize
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} images from {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, class_names

def train_model(data_dir="fruit_dataset", epochs=50, batch_size=32):
    """
    Train the fruit classification model
    """
    # Load and preprocess data
    X_train, X_val, y_train, y_val, class_names = load_and_preprocess_data(data_dir)
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, len(class_names))
    y_val_cat = tf.keras.utils.to_categorical(y_val, len(class_names))
    
    # Create model
    model = create_cnn_model(num_classes=len(class_names))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_fruit_model.keras', save_best_only=True)
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('fruit_classifier_model.keras')
    print("\nModel saved as 'fruit_classifier_model.keras'")
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    return model, history, class_names

def load_model_and_classes(model_path='fruit_classifier_model.keras', classes_path='class_names.txt'):
    """
    Load trained model and class names
    """
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Load class names
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = FRUIT_CLASSES  # Use default
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for prediction
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_fruit(model, image_path, class_names):
    """
    Predict fruit type from image
    """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image, verbose=0)
    
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_fruit = class_names[predicted_class_index]
    
    return predicted_fruit, confidence, predictions[0]

def predict_and_display(model, image_path, class_names):
    """
    Predict and display the result with image
    """
    fruit_name, confidence, all_predictions = predict_fruit(model, image_path, class_names)
    
    # Display image and predictions
    img = Image.open(image_path)
    plt.figure(figsize=(12, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Show predictions
    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, all_predictions)
    plt.title('Prediction Confidence')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    
    # Highlight the predicted class
    max_idx = np.argmax(all_predictions)
    bars[max_idx].set_color('red')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🍎 Predicted Fruit: {fruit_name.upper()}")
    print(f"🎯 Confidence: {confidence:.2%}")
    print(f"📊 All Predictions:")
    for i, (fruit, conf) in enumerate(zip(class_names, all_predictions)):
        print(f"   {fruit}: {conf:.3f} ({conf*100:.1f}%)")
    
    return fruit_name, confidence

def main():
    """
    Main function to train or predict
    """
    print("🍓 Fruit Classification System 🍓")
    print("=" * 40)
    
    # Check if model exists
    if os.path.exists('fruit_classifier_model.keras'):
        print("✅ Model found! Loading for prediction...")
        model, class_names = load_model_and_classes()
        
        if model is not None:
            print(f"📁 Classes: {class_names}")
            
            # Select image for prediction
            root = tk.Tk()
            root.withdraw()
            
            print("\n📸 Please select a fruit image to classify...")
            image_path = filedialog.askopenfilename(
                title="Select a fruit image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )
            
            if image_path:
                print(f"Selected: {os.path.basename(image_path)}")
                predict_and_display(model, image_path, class_names)
            else:
                print("❌ No image selected!")
        
    else:
        print("❌ No trained model found!")
        
        # Check if dataset exists
        if os.path.exists('fruit_dataset'):
            response = input("\n🤔 Would you like to train a new model? (y/n): ")
            if response.lower() == 'y':
                print("\n🚀 Starting training process...")
                model, history, class_names = train_model()
                print("✅ Training completed!")
            else:
                print("👋 Goodbye!")
        else:
            print("\n📁 Please create a 'fruit_dataset' folder with your fruit images organized like this:")
            print("fruit_dataset/")
            print("├── apple/")
            print("│   ├── apple1.jpg")
            print("│   └── apple2.jpg")
            print("├── banana/")
            print("├── grapes/")
            print("├── kiwi/")
            print("├── mango/")
            print("└── strawberry/")

if __name__ == "__main__":
    main()