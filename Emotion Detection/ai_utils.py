import threading
import pyttsx3
import google.generativeai as genai

# --- GEMINI & TTS UTILITIES ---

def get_gemini_response(emotion: str) -> str:
    """Generates a text response based on the detected emotion using the Gemini API."""
    prompt_map = {
        "happy": "The user looks happy. Say something cheerful and encouraging.",
        "sad": "The user looks sad. Offer a simple, comforting, and encouraging statement.",
        "angry": "The user looks angry. Respond with empathy and a calm, brief suggestion to take a deep breath.",
        "neutral": "The user looks neutral. Start a friendly, open-ended chat about their day.",
        "surprise": "The user looks surprised. React with a brief, excited question about what happened.",
        "fear": "The user looks scared. Offer a brief, reassuring message.",
        "disgust": "The user looks disgusted. Briefly comment on a funny face and ask what they reacted to."
    }
    
    # Use 'neutral' or a default if the specific emotion isn't in the map
    prompt = prompt_map.get(emotion, "Say something friendly and brief.") 
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        print(f"Gemini response: {response.text}")
        return response.text.strip() if response.text else "I'm here for you."
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "I am having trouble connecting right now, but I'm listening."

def text_to_speech_thread(text: str, engine: pyttsx3.Engine):
    """Plays a given text using pyttsx3 in a separate thread."""
    engine.say(text)
    engine.runAndWait()

def speak_async(text: str, engine: pyttsx3.Engine) -> threading.Thread:
    """
    Creates and starts a new thread for text-to-speech.
    Returns the thread object.
    """
    voice_thread = threading.Thread(
        target=text_to_speech_thread, 
        args=(text, engine), 
        daemon=True
    )
    voice_thread.start()
    return voice_thread