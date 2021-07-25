# 3D screen demo

This is a simple project to test using the webcam and face detection to try simulate depth on a normal 2D screen.
Steps involved:

*   Draw a square in front of another one.
*   Move the front square based on the position of the viewer's face.
*   Amount it moves depends on the depth between the squares.


![png](https://github.com/stanton119/data-analysis/blob/master/machine_vision/3d_screen/demo.png)

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
