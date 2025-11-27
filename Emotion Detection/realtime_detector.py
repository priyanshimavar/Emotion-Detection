import cv2
import numpy as np
import pyttsx3
import threading
import time # <-- New import for cooldown timer
from keras.models import Sequential
from typing import Optional

# Assume modules are in the same directory
from config import EMOTION_DICT
from ai_utils import get_gemini_response, speak_async
from model_utils import load_model 

# --- REAL-TIME DETECTION (FUNCTION) ---
def run_real_time_detection():
    """
    Loads the saved model and runs real-time emotion detection with camera feed.
    
    Includes a cooldown timer to limit Gemini API calls and manage quotas.
    """
    try:
        emotion_model = load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 1. Setup
    engine = pyttsx3.init()
    
    # Configure TTS voice (optional)
    voices = engine.getProperty('voices')
    try:
        # Index 1 is often a clear voice, but may vary by system.
        engine.setProperty('voice', voices[1].id) 
        engine.setProperty('rate', 160)
    except IndexError:
        print("Could not set specific voice, using default.")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible. Check hardware and permissions.")
        return

    # State variables for chat/voice control
    last_emotion: Optional[str] = None
    chat_response: str = "Hello! I'm listening. Show me how you feel." # Initial message
    voice_thread: Optional[threading.Thread] = None

    # --- COOLDOWN IMPLEMENTATION ---
    # Set a generous cooldown to respect the 50 calls/day Free Tier limit.
    # 10 seconds means max 6 calls/minute, max 360 calls/hour.
    # Let's use 10 seconds for a natural conversation flow and better quota management.
    COOLDOWN_SECONDS = 10 
    
    # Initialize to allow an immediate first response
    last_response_time = time.time() - COOLDOWN_SECONDS 
    
    print("Starting real-time video feed. Press 'q' to quit.")

    # 2. Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Resize for better visualization (optional)
        frame = cv2.resize(frame, (1280, 720)) 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        num_faces = face_detector.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5
        )

        for (x, y, w, h) in num_faces:
            # ... (Image processing and prediction remains the same) ...
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray_frame, (48, 48))
            cropped_img = cropped_img / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=-1)
            cropped_img = np.expand_dims(cropped_img, axis=0)

            emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(emotion_prediction))
            predicted_emotion = EMOTION_DICT.get(maxindex, 'unknown')

            # 3. Chat and TTS Logic (Only on emotion change AND after cooldown)
            
            emotion_changed = predicted_emotion != last_emotion
            cooldown_expired = (time.time() - last_response_time) >= COOLDOWN_SECONDS
            
            if emotion_changed and cooldown_expired:
                
                # --- UPDATE THE COOLDOWN TIMER ---
                last_response_time = time.time()
                last_emotion = predicted_emotion
                
                # Get response from Gemini
                chat_response = get_gemini_response(predicted_emotion)
                
                # Start new TTS thread, passing the required 'engine' argument
                voice_thread = speak_async(chat_response, engine)

            # 4. Display results on frame
            # Display current predicted emotion
            cv2.putText(frame, predicted_emotion, (x + 5, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display chat response
            cv2.putText(frame, chat_response, (x + 5, y + h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 5. Show frame and check for quit command
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")