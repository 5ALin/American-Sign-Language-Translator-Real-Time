import flet as ft
import json
import subprocess
import os
import sys


# File to store user settings
CONFIG_FILE = "user_config.json"

# List of all landmark names
LANDMARKS = [
    "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer",
    "Right Eye Inner", "Right Eye", "Right Eye Outer",
    "Left Ear", "Right Ear", "Mouth Left", "Mouth Right",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky",
    "Left Index", "Right Index", "Left Thumb", "Right Thumb",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index"
]

def load_config():
    """Load selected landmarks from the config file (if exists)."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        return data.get("selected_landmarks", [])
    return []

def save_config(selected_landmarks):
    """Save the selected landmarks to a config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"selected_landmarks": selected_landmarks}, f)

def start_tracking(e):
    """Save selected landmarks and start `main.py`."""
    # Save the selected landmarks from the GUI checkboxes
    selected_landmarks = [checkbox.label for checkbox in landmark_checkboxes if checkbox.value]
    save_config(selected_landmarks)  # Save the selection to user_config.json

    # Get the path to Python executable and `main.py`
    python_executable = sys.executable  # Dynamically uses the current Python environment
    main_script = os.path.join(os.path.dirname(__file__), "main.py")  # Adjusts to the script's location

    # Check if `main.py` exists
    if not os.path.exists(main_script):
        print(f"Error: Could not find main.py at {main_script}")
        return

    try:
        # Launch `main.py` as a detached process
        subprocess.Popen(
            [python_executable, main_script],  # Command to run `main.py`
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Create a new terminal window
        )
        print("Main.py started successfully.")  # Confirm success
    except Exception as ex:
        print(f"Error launching main.py: {ex}")  # Log any errors



def main(page: ft.Page):
    """Main GUI setup function."""
    page.title = "Landmark Tracker Settings"
    page.window_width = 400
    page.window_height = 600
    page.scroll = "adaptive"

    global landmark_checkboxes
    saved_landmarks = load_config()

    # Create checkboxes for all landmarks (restore saved selections)
    landmark_checkboxes = [
        ft.Checkbox(label=landmark, value=(landmark in saved_landmarks)) for landmark in LANDMARKS
    ]

    # Scrollable list of landmarks
    scrollable_landmarks = ft.ListView(
        controls=landmark_checkboxes, expand=True, spacing=10
    )

    # Start button
    start_button = ft.ElevatedButton("Start Tracking", on_click=start_tracking)

    # Layout
    page.add(ft.Text("Select Landmarks to Track", size=20, weight="bold"))
    page.add(scrollable_landmarks)
    page.add(start_button)

ft.app(target=main)
