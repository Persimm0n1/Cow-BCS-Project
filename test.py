import requests
import os

# --- 1. CONFIGURE YOUR TEST ---

# The URL of your locally running API
API_URL = "http://127.0.0.1:8080/predict" 

# The path to the cow image you want to test
IMAGE_PATH = "path/to/your/test_cow.jpg" # <--- IMPORTANT: Change this!

# --- 2. RUN THE TEST ---

def test_prediction():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Test image not found at {IMAGE_PATH}")
        return

    print(f"Sending {IMAGE_PATH} to {API_URL}...")

    try:
        # Open the image file in binary-read mode
        with open(IMAGE_PATH, 'rb') as f:
            
            # 'files' is how you send a file upload in a POST request
            files = {
                'image': (os.path.basename(IMAGE_PATH), f, 'image/jpeg')
            }
            
            # Send the request
            response = requests.post(API_URL, files=files, timeout=30)
            
            # --- 3. PRINT THE RESULTS ---
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                print("--- SUCCESS ---")
                print(response.json())
            else:
                print("--- FAILED ---")
                print("API returned an error:")
                try:
                    print(response.json())
                except:
                    print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nConnection Error: Could not connect to {API_URL}")
        print("Are you sure your 'app.py' server is running in another terminal?")
        print(f"Details: {e}")

if __name__ == "__main__":
    test_prediction()