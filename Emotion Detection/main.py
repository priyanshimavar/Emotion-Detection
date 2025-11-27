from realtime_detector import run_real_time_detection
from model_utils import train_and_save_model
from config import TRAIN_DIR, TEST_DIR 

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    print("Welcome to the Real-Time Emotion Detection Chatbot!")
    print("-" * 50)
    
    # Simple menu for the user
    print("Select an option:")
    print("1. Run Real-Time Detection (Requires pre-trained model)")
    print("2. Train and Save Model (Requires dataset and will take time)")
    print("-" * 50)
    
    choice = input("Enter your choice (1 or 2): ")

    if choice == '2':
        # You can increase epochs for better training
        try:
            model, history, emotion_map = train_and_save_model(
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                epochs=10 # Example: set to 10 epochs for better results
            )
            print("Training complete. You can now run the detector (option 1).")
        except FileNotFoundError:
            print("ERROR: Training dataset paths in config.py are incorrect. Please update TRAIN_DIR and TEST_DIR.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
            
    elif choice == '1':
        run_real_time_detection()
        
    else:
        print("Invalid choice. Please enter '1' or '2'.")