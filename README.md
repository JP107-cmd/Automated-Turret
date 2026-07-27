# Simple Face Tracker

A small Python script that connects to your MacBook camera and tracks faces in real
time using OpenCV's DNN face detector (ResNet-10 SSD). For each detected face it draws a
green box with a **persistent ID** that stays with the face as it moves, plus the
confidence score. It also overlays the total face count and a live **FPS counter**.

The persistent IDs come from a lightweight centroid tracker built into the script: faces
are matched frame-to-frame by nearest box-center, so each person keeps the same "ID N"
label instead of being re-numbered every frame.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
cd "Simple face-tracker"
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### 2. Download the model

The detector needs two pre-trained files in a `models/` folder:

```bash
mkdir -p models
curl -L -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

## Run

```bash
venv/bin/python face_tracker.py
```

A window opens showing your camera feed with face boxes. Press **`q`** or **`ESC`** to quit.

### Options

```bash
venv/bin/python face_tracker.py --confidence 0.6   # raise the detection threshold (default 0.5)
venv/bin/python face_tracker.py --camera 1         # use a different camera index (default 0)
```

## Arduino turret output

The script can drive a pan/tilt servo turret. Each frame it picks a target face, converts
its position into **pan/tilt angles**, and streams them to an Arduino over USB serial as
one text line:

```
pan,tilt\n      e.g.  90,75      (two integers, each 0-180 degrees)
```

The Arduino parses that line and drives two servos. The matching sketch is in
[turret_arduino/turret_arduino.ino](turret_arduino/turret_arduino.ino); beyond a bare
`servo.write()` it adds **slew-rate limiting** (servos ease toward each target for smooth
motion) and a **failsafe** that recenters the turret if commands stop arriving.

**Works without hardware as well** With no `--serial-port`, the script prints the command
stream to the console (and shows the live aim on the video overlay as `AIM P:.. T:..`),
so you can watch the angles change as you move:

```bash
venv/bin/python face_tracker.py
```

**With an Arduino connected**, pass its port (find it with `ls /dev/cu.usbmodem*`):

```bash
venv/bin/python face_tracker.py --serial-port /dev/cu.usbmodem1101
```

### Turret options

| Flag | Purpose |
|---|---|
| `--mount turret` *(default)* | Camera is **on** the turret → proportional centering control. |
| `--mount fixed` | Camera is **stationary** → maps face position straight to an angle. |
| `--mount offset` | Stationary camera at a **known mm offset** from the turret → full parallax + depth correction (see below). |
| `--hfov 55` | Camera horizontal field of view, degrees (for `offset`). |
| `--face-width 160` | Assumed real face width in mm, for depth estimation (for `offset`). |
| `--track-id N` | Lock onto a specific face ID instead of the largest face. |
| `--invert-pan` / `--invert-tilt` | Flip a direction if the turret moves the wrong way. |
| `--baud 115200` | Serial baud rate (must match the sketch). |

> **Choosing `--mount`:** if the camera moves with the servos, keep the default
> `turret`. If the camera is bolted to a desk and only the turret moves, use `fixed`
> (simple) or `offset` (accurate) — otherwise the proportional loop never sees the face
> re-center and will drive to a limit.

### `--mount offset`: correcting for camera/turret separation

Because the turret isn't in the same spot as the camera, the angle the camera *sees* a
face at isn't the angle the turret must *point* at — that's parallax. The `offset` mode
corrects for it:

1. The camera FOV (`--hfov`) gives a focal length in pixels.
2. The face's apparent width gives a rough **distance** (`--face-width` is the assumed
   real width; bigger box = closer).
3. From distance + pixel position we reconstruct the face's 3D position relative to the
   camera, subtract the turret's offset, and take the resulting bearing/elevation.

The turret offset is set in [face_tracker.py](face_tracker.py)
(**For my computer its 150 mm in front of** and **135 mm below, but you can change the constants as you wish** the camera):

```python
TURRET_OFFSET_RIGHT_MM = 0.0      # +ve = turret to the camera's right
TURRET_OFFSET_UP_MM = -135.0      # 135 mm below the camera
TURRET_OFFSET_FORWARD_MM = 150.0  # 150 mm in front of the camera
```

> The distance estimate from a single camera is approximate, so calibrate `--hfov` and
> `--face-width` for best accuracy. The closer the target, the more the offset matters.

## macOS camera permission

This script is made for Mac OS. The first run triggers a macOS camera-permission prompt 
for the app that launches the script (Terminal, iTerm, or VS Code). Allow it. If no prompt 
appears or you denied it, enable camera access manually under **System Settings → 
Privacy & Security → Camera**, then restart your terminal/IDE and run again.
