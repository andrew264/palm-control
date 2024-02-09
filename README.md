# palm-control

This is a simple python script that uses the `mediapipe` library to detect the hand and fingers and control the mouse
pointer using the palm and fingers.

# TODO (what I think I need to do)

- [x] use `mediapipe` to get hand cords (easy)
- [x] use `Kalman filtering` to smooth the outputs (kinda easy)
- [x] map them fingers to mouse cursor movements (the hard part) kainda meh
- [ ] map clicks and drags (why is this so hard)
- [ ] using something like a time-series model to find future outputs (high polling rate) ?? (Should be doable?)
- [ ] profit ??