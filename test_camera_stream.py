#!/usr/bin/env python3
"""Test script to verify camera streaming works end-to-end."""

import time
import cv2
from vision.stream import DepthAIFrameStream, StreamConfig

config = StreamConfig(resolution=(1280, 720), fps=30.0)
stream = DepthAIFrameStream(config)

print("Starting camera stream...")
stream.start()

print("Waiting for frames...")
for i in range(100):
    frame, _ = stream.get_latest()
    if frame is not None:
        print(f"Frame {i}: Got frame with shape {frame.shape}")
        if i == 0:
            cv2.imwrite("/tmp/test_frame.jpg", frame)
            print("Saved test frame to /tmp/test_frame.jpg")
    else:
        print(f"Frame {i}: No frame yet")
    time.sleep(0.1)

print("Stopping camera stream...")
stream.stop()
print("Done!")
