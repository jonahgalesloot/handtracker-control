**Hand Tracking & Handwriting OCR**

- **Purpose:** Real-time hand-tracking cursor + handwriting input using MediaPipe and EasyOCR. Captures hand landmarks from the camera, maps index finger motion to cursor movement, supports click/scroll gestures, and converts drawn handwriting into text.

**Included Files**

- `hand_control.py`: Main application — reads landmark messages (ZeroMQ) from the tracker process, renders overlay, handles gestures, runs EasyOCR on drawn strokes, and types recognized text.
- `handtracker_module.py`: Camera capture + MediaPipe hand landmark extraction; publishes frames + landmark JSON over ZeroMQ.
- `requirements.txt`: Packages needed to reproduce the working environment (use this to recreate the venv).

**Prerequisites**

- Windows machine (tested on Windows 11)
- Python 3.10 (recommended) — MediaPipe wheels are Python-version specific
- A working camera

**Quick Setup**

1. Clone or download this repository.
2. Create a fresh venv and activate it (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

**Run the program**

Run the main script which launches the tracker subprocess automatically:

```powershell
python hand_control.py
```

**Controls**
- `Index touch middle finger`: Causes cursor to follow index finger
- `Index touch thumb`: Triggers a cursor click
- `Middle finger touch thumb`: Scrolls based on index angle; up for scroll up and sideways for scroll down

**Keyboard Shortcuts**

- `Ctrl+Alt+D`: Toggle the overlay display on/off.
- `Ctrl+Alt+Q`: Exit the program (requests a clean shutdown).

**How handwriting input works**

- Put your ring finger and thumb together to enter drawing mode. The program records index-tip positions while that gesture is held, builds a stroke image, preprocesses it, and runs EasyOCR to recognize characters.
- The recognized character is typed automatically via `pyautogui`.



-- End of README --