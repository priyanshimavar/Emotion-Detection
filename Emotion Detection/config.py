import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- GLOBAL VARIABLES & CONFIGURATION ---
load_dotenv()
try:
    # Configure GenAI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"GenAI configuration failed. Ensure GOOGLE_API_KEY is set in your .env file. Error: {e}")

# Global dictionary to map model output indices to emotion names.
# This should ideally be loaded from a saved file after training,
# but is hardcoded here for the real-time script's initial load.
EMOTION_DICT = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# File paths
MODEL_JSON_PATH = "emotion_model.json"
MODEL_WEIGHTS_PATH = "emotion_model.weights.h5"

# Training directories (Update these if you run training)
TRAIN_DIR = r"C:\Users\x\OneDrive\Desktop\Emotion-Detection model\2ndDataset\train"
TEST_DIR = r"C:\Users\x\OneDrive\Desktop\Emotion-Detection model\2ndDataset\test"