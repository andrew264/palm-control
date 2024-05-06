sudo modprobe v4l2loopback exclusive_caps=1 card_label="Scrcpy Webcam"
scrcpy --video-source=camera --no-audio  --camera-size=1920x1080 --camera-id=0 --camera-fps=60 --v4l2-sink=/dev/video0 --no-video-playback
