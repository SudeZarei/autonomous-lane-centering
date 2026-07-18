# Autonomous Lane Tracking Robot

This repository contains the source code and hardware configuration for an autonomous mobile robot capable of real-time lane detection and navigation. The project utilizes a Raspberry Pi 3 Model B and computer vision algorithms to process camera feeds and drive an Ackermann steering mechanism. 

The system is designed to run efficiently on resource-constrained hardware without relying on computationally heavy deep learning models.

## Features

* Real-Time Vision Processing: Captures frames via the PiCamera2 interface, applies HLS color space conversion, and uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to mitigate lighting variations and shadows.
* Dynamic Lane Segmentation: Uses color thresholding and morphological operations (Closing) to isolate the road and lane markings. The algorithm features dynamic background clipping to ignore reflections from non-road surfaces.
* Ackermann Steering Control: Processes contour area ratios (Left vs. Right) to determine the steering angle. Controls the steering servo and DC drive motor via hardware-timed PWM using the `pigpio` library for jitter-free movement.
* Web-Based Monitoring & Headless Operation: Integrates a customized version of [CamUI](https://github.com/monkeymademe/CamUI) to run the Raspberry Pi without a desktop GUI. The Flask-based server streams the processed video feed, allowing real-time visualization of the algorithm's frame transformations and navigation decisions over a local network.

## Hardware Components

* Raspberry Pi 3 Model B (Running headless OS)
* 5MP Raspberry Pi Camera Module (OV5647)
* Custom RC car chassis with Ackermann steering
* MG90S Micro Servo (for steering)
* Mini L293D Motor Driver (for the rear DC motor)
* Parallel Li-ion battery pack with a Type-C Boost Charger (5V output)

## Software Stack

* Python 3
* OpenCV (Image Processing)
* Flask (Web Interface)
* Pigpio (Hardware PWM control)
* PiCamera2 (Camera Interface)

## Setup and Execution

1. Hardware Setup: Ensure the servo is connected to GPIO 18, and the motor driver pins (IN1, IN2, ENA) are connected to GPIO 24, 23, and 25 respectively.
2. Start the pigpio daemon:
   ```bash
   sudo pigpiod
   ```
3. Install the required Python packages:
   ```bash
   pip install opencv-python-headless flask numpy
   ```
4. Run the main application:
   ```bash
   python3 app.py
   ```

## Acknowledgments

The web interface and camera streaming infrastructure are built upon [CamUI](https://github.com/monkeymademe/CamUI). The original source code was modified and extended to integrate the custom `LaneDetector` computer vision pipeline, enabling live debugging of the robot's logic while operating headlessly.
