# Face distance

This is a simple project to estimate the distance of a face to the webcam.
This can be used to detect when a person's neck posture becomes poor when using a laptop.

Steps involved:

*   Estimate eye position.
*   Get distance between eyes.
*   Plot on the image

You can see the white bar at the top which reflects the eye distance
![jpeg](https://github.com/stanton119/data-analysis/raw/master/machine_vision/face_distance/face_proc.jpeg)

## Installation
Env setup:
```
python3 -m venv machine_vision_env
source machine_vision_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run demo:
```
source machine_vision_env/bin/activate
python live_demo.py
```
