import os
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay # type: ignore
from typing import Tuple, Dict
from tensorflow.keras.callbacks import History # type: ignore

# Assume config.py is in the same directory
from config import MODEL_JSON_PATH, MODEL_WEIGHTS_PATH 


def build_emotion_model(input_shape: tuple, num_classes: int) -> Sequential:
    """Defines the CNN model architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_save_model(train_dir: str, test_dir: str, epochs: int = 2) -> Tuple[Sequential, History, Dict[int, str]]:
    """
    Trains, saves the emotion detection model, and returns the emotion dictionary.
    """
    print("Starting model training...")
    
    # 1. Data Generators
    train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = validation_data_gen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
    
    # Map from class index (integer) to emotion name (string)
    emotion_dict = {v: k for k, v in train_generator.class_indices.items()}

    # 2. Model Building and Compilation
    emotion_model = build_emotion_model(
        input_shape=(48, 48, 1), 
        num_classes=train_generator.num_classes
    )

    lr_schedule = ExponentialDecay(0.0001, decay_steps=100000, decay_rate=0.96)
    optimizer = Adam(learning_rate=lr_schedule)

    emotion_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # 3. Training
    emotion_model_info = emotion_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # 4. Saving the model
    model_json = emotion_model.to_json()
    with open(MODEL_JSON_PATH, "w") as json_file:
        json_file.write(model_json)
    emotion_model.save_weights(MODEL_WEIGHTS_PATH)
    
    print(f"Model trained and saved successfully to {MODEL_JSON_PATH} and {MODEL_WEIGHTS_PATH}.")

    return emotion_model, emotion_model_info, emotion_dict

def load_model() -> Sequential:
    """Loads the emotion detection model from saved files."""
    if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model files not found. Please train the model first. "
            f"Missing: {MODEL_JSON_PATH} or {MODEL_WEIGHTS_PATH}"
        )
    
    with open(MODEL_JSON_PATH, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model loaded successfully.")
    
    return emotion_model