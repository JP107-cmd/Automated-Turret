/*
 * Face-tracker turret receiver.
 *
 * Listens on USB serial for lines of the form "pan,tilt\n" (each 0-180 degrees),
 * e.g. "90,75", and points two servos accordingly. This pairs with face_tracker.py,
 * which computes and streams those angles from the detected face position (including
 * parallax/offset correction in --mount offset mode).
 *
 * Two niceties beyond a bare servo.write():
 *   - Slew-rate limiting: the servos ease toward each new target a few degrees at a
 *     time instead of snapping, so motion looks smooth and draws less peak current.
 *   - Failsafe: if no command arrives for a while (camera/script stopped), the turret
 *     returns to center rather than freezing aimed at the last face.
 *
 * Wiring:
 *   - Pan servo  signal -> pin 9
 *   - Tilt servo signal -> pin 10
 *   - Power the servos from a SEPARATE 5-6V supply (NOT the Arduino 5V pin) and tie
 *     that supply's ground to the Arduino ground (common ground). Servos draw enough
 *     current to brown out / reset the board if powered from the 5V pin.
 */

#include <Servo.h>

const int PAN_PIN = 2;
const int TILT_PIN = 3  ;
const long BAUD = 115200;  // must match --baud in face_tracker.py

// Smoothness: max degrees to move per step, and how often to step.
const int MAX_STEP = 2;             // degrees per update (lower = smoother/slower)
const unsigned long STEP_MS = 15;   // step interval in milliseconds

// Failsafe: if no command for this long, recenter to 90/90.
const unsigned long FAILSAFE_MS = 1000;

Servo panServo;
Servo tiltServo;

int panTarget = 90, tiltTarget = 90;  // latest commanded angles
int panNow = 90, tiltNow = 90;        // current (slew-limited) angles

char buffer[16];
byte idx = 0;
unsigned long lastCommandMs = 0;
unsigned long lastStepMs = 0;

// Move `current` toward `target` by at most `maxStep` degrees.
int stepToward(int current, int target, int maxStep) {
  if (target > current) return min(current + maxStep, target);
  if (target < current) return max(current - maxStep, target);
  return current;
}

void setup() {
  Serial.begin(BAUD);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panNow);
  tiltServo.write(tiltNow);
  lastCommandMs = millis();
}

void loop() {
  // 1) Read any complete "pan,tilt" lines and update the targets.
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      buffer[idx] = '\0';
      int p, t;
      if (sscanf(buffer, "%d,%d", &p, &t) == 2) {
        panTarget = constrain(p, 0, 180);
        tiltTarget = constrain(t, 0, 180);
        lastCommandMs = millis();
      }
      idx = 0;  // ready for the next line
    } else if (idx < sizeof(buffer) - 1) {
      buffer[idx++] = c;
    }
    // else: overflow -> drop chars until the next newline resets us
  }

  // 2) Failsafe: lost the feed -> return to center.
  if (millis() - lastCommandMs > FAILSAFE_MS) {
    panTarget = 90;
    tiltTarget = 90;
  }

  // 3) Ease the servos toward their targets for smooth motion.
  if (millis() - lastStepMs >= STEP_MS) {
    lastStepMs = millis();
    panNow = stepToward(panNow, panTarget, MAX_STEP);
    tiltNow = stepToward(tiltNow, tiltTarget, MAX_STEP);
    panServo.write(panNow);
    tiltServo.write(tiltNow);
  }
}
