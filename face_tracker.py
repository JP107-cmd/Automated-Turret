"""Simple real-time face tracker.

Connects to the MacBook camera and uses OpenCV's DNN face detector
(ResNet-10 SSD, Caffe) to draw a bounding box around each detected face,
overlaying a persistent per-face ID, confidence score, the total face count,
and a live FPS counter.

Persistent IDs are assigned by a lightweight centroid tracker: each face keeps
the same "ID N" label as it moves around the frame, instead of being treated as
a brand-new detection every frame.

Turret output: when a target face is present, the script converts its position into
pan/tilt servo angles and streams them to an Arduino over USB serial as a single text
line, e.g. "90,75\n" (pan,tilt in degrees, 0-180). The Arduino just does
servo.write(pan)/servo.write(tilt). See turret_arduino/turret_arduino.ino and the README.

Quit with `q` or `ESC`.
"""

import argparse
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

try:
    import serial  # pyserial; only needed when an Arduino is connected
except ImportError:  # keep the tracker usable even if pyserial isn't installed
    serial = None

# --- Paths to the pre-trained model files (see README for download step) ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
PROTOTXT_PATH = MODELS_DIR / "deploy.prototxt"
WEIGHTS_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# --- Detection settings ---
DEFAULT_CONFIDENCE = 0.5  # minimum confidence to count something as a face
DEFAULT_CAMERA_INDEX = 0  # 0 = built-in FaceTime camera
INPUT_SIZE = (300, 300)  # the network's expected input size
MEAN_SUBTRACTION = (104.0, 177.0, 123.0)  # mean values the model was trained with

# --- Tracking settings ---
# How many consecutive frames a face can go undetected before its ID is dropped.
MAX_DISAPPEARED = 30
# Max pixel distance between centroids to still count as "the same face".
MAX_MATCH_DISTANCE = 80
# How strongly to smooth the FPS readout (0-1; higher = smoother but laggier).
FPS_SMOOTHING = 0.9

# --- Turret / servo settings ---
DEFAULT_BAUD = 115200          # must match Serial.begin() in the Arduino sketch
# Allowed servo angle range and the neutral/center angle (degrees).
PAN_MIN, PAN_MAX, PAN_CENTER = 0, 180, 90
TILT_MIN, TILT_MAX, TILT_CENTER = 0, 180, 90
# "turret" mode (camera mounted ON the turret): proportional closed-loop control.
# Degrees to nudge per frame at full error; deadzone ignores tiny errors near center.
TRACK_GAIN = 10.0
DEADZONE = 0.08

# Default servo direction flips, set to match this rig. Pan runs reversed (the turret
# was turning the wrong way), so flip it by default; override with --no-invert-pan.
DEFAULT_INVERT_PAN = True
DEFAULT_INVERT_TILT = False

# --- Camera + turret geometry (used by --mount offset) ---
# Horizontal field of view of the camera in degrees (calibrate for your camera;
# the MacBook FaceTime cam is ~55 deg). Needed to turn pixels into real-world angles.
CAMERA_HFOV_DEG = 55.0
# Real width of a face including a little detector-box padding, in mm. Used to estimate
# distance from the camera ("bigger box = closer"). Calibrate for best results.
REAL_FACE_WIDTH_MM = 160.0
# Where the turret sits relative to the camera (camera = origin), in millimetres.
# Defaults match the user's rig: 150 mm in front of and 135 mm below the camera.
TURRET_OFFSET_RIGHT_MM = 0.0      # +ve = turret is to the camera's right
TURRET_OFFSET_UP_MM = -135.0      # turret is 135 mm DOWN from the camera
TURRET_OFFSET_FORWARD_MM = 150.0  # turret is 150 mm in FRONT of the camera
# Clamp for the (rough) single-camera depth estimate so bad detections can't fling the aim.
MIN_DEPTH_MM, MAX_DEPTH_MM = 200.0, 5000.0

WINDOW_NAME = "Face Tracker (press q or ESC to quit)"
BOX_COLOR = (0, 255, 0)      # green (BGR) - untargeted faces
TARGET_COLOR = (0, 165, 255)  # orange - the face the turret is aiming at
TEXT_COLOR = (0, 255, 0)


def clamp(value, low, high):
    """Constrain a value to the inclusive [low, high] range."""
    return max(low, min(high, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face tracker using OpenCV DNN.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence, 0-1 (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index to open (default: {DEFAULT_CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help="Serial port of the Arduino, e.g. /dev/cu.usbmodem1101. "
        "If omitted, pan/tilt commands are printed to the console instead of sent.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate; must match the Arduino sketch (default: {DEFAULT_BAUD}).",
    )
    parser.add_argument(
        "--mount",
        choices=("turret", "fixed", "offset"),
        default="turret",
        help="Camera mounting. 'turret' = camera moves with the servos "
        "(proportional centering); 'fixed' = stationary camera, maps face position "
        "straight to an angle; 'offset' = stationary camera at a known mm offset from "
        "the turret, with full parallax + depth correction. Default: turret.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=CAMERA_HFOV_DEG,
        help=f"Camera horizontal field of view in degrees, for --mount offset "
        f"(default: {CAMERA_HFOV_DEG}).",
    )
    parser.add_argument(
        "--face-width",
        type=float,
        default=REAL_FACE_WIDTH_MM,
        help=f"Assumed real face width in mm, for depth estimation in --mount offset "
        f"(default: {REAL_FACE_WIDTH_MM}).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Lock onto a specific face ID. If omitted, aims at the largest face.",
    )
    parser.add_argument(
        "--invert-pan",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INVERT_PAN,
        help="Flip pan direction if the turret turns the wrong way "
        "(default: %(default)s; disable with --no-invert-pan).",
    )
    parser.add_argument(
        "--invert-tilt",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INVERT_TILT,
        help="Flip tilt direction if the turret tilts the wrong way "
        "(default: %(default)s; disable with --no-invert-tilt).",
    )
    return parser.parse_args()


def load_network() -> cv2.dnn.Net:
    """Load the Caffe face-detection network, exiting with guidance if files are missing."""
    missing = [p for p in (PROTOTXT_PATH, WEIGHTS_PATH) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        sys.exit(
            "Error: missing model file(s):\n  "
            f"{names}\n\n"
            "Download them by following the 'Download the model' step in README.md."
        )
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(WEIGHTS_PATH))


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the camera, exiting with a helpful message if it can't be accessed."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"Error: could not open camera index {index}.\n"
            "On macOS, grant camera access to your terminal/IDE under\n"
            "System Settings -> Privacy & Security -> Camera, then try again."
        )
    return cap


class CentroidTracker:
    """Assigns a stable integer ID to each face and follows it across frames.

    The idea is simple: every detection has a centroid (box center). Each frame we
    match new centroids to the ones we already know by nearest distance. A matched
    centroid keeps its existing ID; an unmatched new centroid gets a fresh ID; and an
    existing ID that goes unseen for too many frames is forgotten.
    """

    def __init__(self, max_disappeared: int = MAX_DISAPPEARED,
                 max_distance: float = MAX_MATCH_DISTANCE):
        self.next_id = 0
        self.objects: OrderedDict[int, tuple] = OrderedDict()  # id -> centroid (x, y)
        self.disappeared: OrderedDict[int, int] = OrderedDict()  # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid) -> int:
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_id += 1
        return object_id

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        """Take this frame's centroids and return a list of IDs, one per centroid.

        The returned list is parallel to `centroids`, so result[i] is the ID assigned
        to centroids[i].
        """
        # No detections this frame: mark everyone as missing, drop the stale ones.
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return []

        # Nothing tracked yet: every centroid is a brand-new face.
        if len(self.objects) == 0:
            return [self._register(c) for c in centroids]

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        input_centroids = np.array(centroids)

        # Distance from every known centroid to every new centroid.
        distances = np.linalg.norm(
            object_centroids[:, None] - input_centroids[None, :], axis=2
        )

        # Greedily match the closest pairs first.
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        assigned_ids = [None] * len(input_centroids)
        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue  # too far apart to be the same face
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            assigned_ids[col] = object_id
            used_rows.add(row)
            used_cols.add(col)

        # Known faces that weren't matched this frame: age them out if stale.
        for row in range(len(object_centroids)):
            if row in used_rows:
                continue
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        # New centroids that didn't match anything: register as new faces.
        for col in range(len(input_centroids)):
            if col not in used_cols:
                assigned_ids[col] = self._register(input_centroids[col])

        return assigned_ids


class TurretController:
    """Turns a target face position into pan/tilt servo angles for the Arduino.

    Two strategies, chosen by `mount`:

    - "turret": the camera is mounted ON the turret, so moving the servos changes the
      view. We use proportional closed-loop control: nudge the angles to push the face
      toward the center of the frame. When the face is centered the error is ~0 and the
      turret holds still.
    - "fixed": the camera is stationary and separate from the turret. We map the face's
      position in the frame directly to an absolute angle (left edge -> one end of travel,
      right edge -> the other).

    Sign conventions depend on how your servos are wired/oriented, so `invert_pan` /
    `invert_tilt` flip the direction if the turret moves the wrong way.
    """

    def __init__(self, mount="turret", invert_pan=False, invert_tilt=False,
                 hfov_deg=CAMERA_HFOV_DEG, real_face_width_mm=REAL_FACE_WIDTH_MM):
        self.mount = mount
        self.invert_pan = invert_pan
        self.invert_tilt = invert_tilt
        self.hfov_deg = hfov_deg
        self.real_face_width_mm = real_face_width_mm
        self.pan = float(PAN_CENTER)
        self.tilt = float(TILT_CENTER)

    def update(self, box, frame_width, frame_height):
        """Update internal pan/tilt from the target face box; return (pan, tilt) ints."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if self.mount == "fixed":
            fx = cx / frame_width   # 0.0 (left) .. 1.0 (right)
            fy = cy / frame_height  # 0.0 (top) .. 1.0 (bottom)
            if self.invert_pan:
                fx = 1.0 - fx
            if self.invert_tilt:
                fy = 1.0 - fy
            self.pan = PAN_MIN + fx * (PAN_MAX - PAN_MIN)
            self.tilt = TILT_MIN + fy * (TILT_MAX - TILT_MIN)

        elif self.mount == "offset":
            pan_deg, tilt_deg = self._aim_with_parallax(
                cx, cy, max(1.0, x2 - x1), frame_width, frame_height
            )
            self.pan = PAN_CENTER + (-pan_deg if self.invert_pan else pan_deg)
            self.tilt = TILT_CENTER + (-tilt_deg if self.invert_tilt else tilt_deg)

        else:  # "turret": proportional centering
            err_x = (cx - frame_width / 2) / (frame_width / 2)    # -1 .. 1
            err_y = (cy - frame_height / 2) / (frame_height / 2)  # -1 .. 1
            if abs(err_x) > DEADZONE:
                step = err_x * TRACK_GAIN
                self.pan += step if self.invert_pan else -step
            if abs(err_y) > DEADZONE:
                step = err_y * TRACK_GAIN
                self.tilt += step if self.invert_tilt else -step

        self.pan = clamp(self.pan, PAN_MIN, PAN_MAX)
        self.tilt = clamp(self.tilt, TILT_MIN, TILT_MAX)
        return int(round(self.pan)), int(round(self.tilt))

    def _aim_with_parallax(self, cx, cy, face_width_px, frame_width, frame_height):
        """Compute pan/tilt angles (deg from center) that point the *turret* at the face.

        Steps:
        1. Convert the camera's field of view into a focal length in pixels.
        2. Estimate the face's distance from the camera using its apparent width
           (a known real width projected through that focal length).
        3. Reconstruct the face's 3D position relative to the camera (X right, Y up,
           Z forward, in mm).
        4. Subtract the turret's offset to get the face position relative to the turret.
        5. Take the bearing (pan) and elevation (tilt) of that vector.
        """
        # 1) Focal length in pixels from the horizontal FOV.
        f_px = (frame_width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

        # 2) Rough monocular depth from the apparent face width, clamped for safety.
        depth = f_px * self.real_face_width_mm / face_width_px
        depth = clamp(depth, MIN_DEPTH_MM, MAX_DEPTH_MM)

        # 3) Face position relative to the camera (mm). Image Y grows downward, so the
        #    upward world axis flips sign.
        px = (cx - frame_width / 2.0) * depth / f_px
        py = -(cy - frame_height / 2.0) * depth / f_px
        pz = depth

        # 4) Vector from the turret to the face = face_pos - turret_offset.
        dx = px - TURRET_OFFSET_RIGHT_MM
        dy = py - TURRET_OFFSET_UP_MM
        dz = pz - TURRET_OFFSET_FORWARD_MM

        # 5) Bearing (left/right) and elevation (up/down) of that vector.
        pan_deg = math.degrees(math.atan2(dx, dz))
        tilt_deg = math.degrees(math.atan2(dy, math.hypot(dx, dz)))
        return pan_deg, tilt_deg

    def command(self):
        """The exact line sent to the Arduino: 'pan,tilt\\n' (e.g. '90,75\\n')."""
        return f"{int(round(self.pan))},{int(round(self.tilt))}\n"


def select_target(faces, ids, track_id):
    """Choose which detected face to aim at; return its index, or None.

    If `track_id` is set, follow that specific ID (and aim at nothing while it's not
    visible). Otherwise aim at the largest face, which is the most stable choice.
    """
    if not faces:
        return None
    if track_id is not None:
        for i, object_id in enumerate(ids):
            if object_id == track_id:
                return i
        return None
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2), _ in faces]
    return int(np.argmax(areas))


def open_serial(port, baud):
    """Open the Arduino serial port, or return None if no port was requested."""
    if not port:
        return None
    if serial is None:
        sys.exit(
            "Error: pyserial is not installed but --serial-port was given.\n"
            "Install it with: venv/bin/pip install pyserial"
        )
    try:
        conn = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as exc:
        sys.exit(f"Error: could not open serial port {port}: {exc}")
    # Opening the port resets most Arduino boards; wait for it to finish booting.
    time.sleep(2.0)
    return conn


def detect_faces(net: cv2.dnn.Net, frame, min_confidence: float):
    """Run the network on a frame and return a list of (box, confidence) tuples."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=INPUT_SIZE, mean=MEAN_SUBTRACTION
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < min_confidence:
            continue
        # Detections are normalized [0, 1]; scale back to frame pixels.
        box = detections[0, 0, i, 3:7] * [width, height, width, height]
        x1, y1, x2, y2 = box.astype(int)
        # Clamp to frame bounds so drawing never goes off-canvas.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        faces.append(((x1, y1, x2, y2), confidence))
    return faces


def draw_overlay(frame, faces, ids, fps, target_idx=None, pan=None, tilt=None) -> None:
    """Draw boxes/IDs, the targeted face, frame-center crosshair, count, FPS, and aim."""
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    # Crosshair at frame center = where the turret is trying to put the target.
    cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    for i, (((x1, y1, x2, y2), confidence), object_id) in enumerate(zip(faces, ids)):
        is_target = i == target_idx
        color = TARGET_COLOR if is_target else BOX_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target else 2)
        label = f"ID {object_id} | {confidence * 100:.1f}%"
        # Put the label just above the box (or just below if near the top edge).
        label_y = y1 - 8 if y1 - 8 > 10 else y2 + 18
        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
        if is_target:
            # Line from frame center to the target shows the aiming error.
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.line(frame, center, face_center, TARGET_COLOR, 1)

    cv2.putText(
        frame, f"Faces: {len(faces)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA,
    )
    aim_text = f"AIM P:{pan} T:{tilt}" if pan is not None else "AIM: no target"
    cv2.putText(
        frame, aim_text, (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TARGET_COLOR, 2, cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    net = load_network()
    cap = open_camera(args.camera)
    tracker = CentroidTracker()
    turret = TurretController(
        mount=args.mount,
        invert_pan=args.invert_pan,
        invert_tilt=args.invert_tilt,
        hfov_deg=args.hfov,
        real_face_width_mm=args.face_width,
    )
    arduino = open_serial(args.serial_port, args.baud)
    if arduino is None:
        print("No --serial-port given: pan/tilt commands will be printed, not sent.")

    fps = 0.0
    last_time = time.perf_counter()
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from camera; stopping.")
                break

            faces = detect_faces(net, frame, args.confidence)

            # Centroid (box center) of each detection, fed to the tracker for IDs.
            centroids = [
                ((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2), _ in faces
            ]
            ids = tracker.update(centroids)

            # Pick a target face and turn its position into pan/tilt angles.
            height, width = frame.shape[:2]
            target_idx = select_target(faces, ids, args.track_id)
            pan = tilt = None
            if target_idx is not None:
                pan, tilt = turret.update(faces[target_idx][0], width, height)
                command = turret.command()
                if arduino is not None:
                    arduino.write(command.encode("ascii"))
                else:
                    # No hardware: echo the instruction stream (throttled to ~10 Hz).
                    now = time.perf_counter()
                    if now - last_print > 0.1:
                        print(f"-> {command.strip()}")
                        last_print = now

            # Update FPS from the time this loop took, exponentially smoothed.
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instant_fps

            draw_overlay(frame, faces, ids, fps, target_idx, pan, tilt)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close()


if __name__ == "__main__":
    main()
