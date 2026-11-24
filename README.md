# Spatial Vision - Android app

Assistive navigation that runs entirely on the phone.

- **ARCore Depth API** - real distance in metres to any obstacle (walls, doors, furniture, people) on the
  left, ahead and right.
- **YOLOv8n** (ONNX Runtime, 320 px) - names the obstacle when it is one of the 80 COCO classes.
- **Voice** (Android text-to-speech) + **parking-sensor vibration** (faster pulses = closer).
- Tap = "where am I". Long-press / SPEAK = voice command: where am I, pause, resume, repeat,
  warn earlier, normal distance, help.

Every push builds the APK with GitHub Actions; the newest APK is on the **Releases -> latest** page.
Needs a phone with ARCore Depth support and "Google Play Services for AR".
