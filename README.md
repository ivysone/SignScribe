# SignScribe

A small desktop app for practising ASL fingerspelling with your webcam. It shows you a letter, you make the handshape, and if you hold it for about a second you get a point.
 
I originally built this as a web page, then rewrote it in Python with OpenCV so it runs as a normal program.
 
## What it does
 
- Picks a random letter (A, S, B, 5, L, V, U, W, Y, I)
- Tracks your hand through the webcam and draws the skeleton on screen
- Guesses which letter you're making
- Fills a ring while you hold the right sign, then moves on to the next one
- Keeps a streak count
Everything runs on your own machine. The video never gets sent anywhere.
 
## Setup
 
You need Python 3.9 to 3.12 (MediaPipe doesn't have builds for other versions) and a webcam.
 
```bash
pip install -r requirements.txt
python signscribe.py
```
 
The first time you run it, it downloads the hand tracking model (about 10 MB) and saves it next to the script. After that it works offline.
 
On a Mac, allow camera access for Terminal when it asks. If you missed the prompt, turn it on under System Settings > Privacy & Security > Camera.
 
## Controls
 
| Key | What it does |
|-----|--------------|
| `s` | skip the current letter |
| `q` | quit (click the video window first so it's focused) |
 
## Tips
 
- Use one hand, palm facing the camera
- Good lighting helps a lot
- Keep your hand a decent distance from the camera so the whole thing is visible
- If it keeps reading the wrong letter, try holding the shape more still
## How it works
 
1. MediaPipe's hand landmarker finds 21 points on your hand each frame.
2. For each finger, I compare how far the fingertip is from the wrist against how far the middle joint is. If the tip is clearly further out, the finger counts as extended. The thumb is checked against the base of the pinky instead, since it moves differently.
3. The pattern of extended fingers (like thumb + pinky = Y) is matched against a list of signs. V and U have the same fingers up, so those are told apart by how far apart the fingertips are.
4. To stop it flickering, a letter only counts if it shows up in at least half of the last 18 frames.
All distances are divided by the hand size (wrist to middle knuckle), so it works whether you're close to the camera or far away.
 
## Files
 
- `signscribe.py` the whole app
- `requirements.txt` dependencies