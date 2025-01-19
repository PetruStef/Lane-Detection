# Lane Detection Project

This repository contains a Python implementation of a lane detection system using OpenCV. The project is designed to process video frames and identify lane lines on the road in real time.

## Features
- **Region of Interest (ROI):** Focuses on a specific area of the image to limit processing to relevant regions.
- **Edge Detection:** Uses Canny edge detection to highlight the edges in the image.
- **Line Detection:** Implements Hough Line Transform to identify lane lines.
- **Line Stabilization:** Smooths detected lines over multiple frames for stability.
- **Visualization:** Combines multiple processing steps into a single visualization frame for better understanding.
  
- ## How It Works

1. **Region of Interest (ROI):**
   - A trapezoidal region is defined to focus the detection on the road area.
2. **Edge Detection:**
   - The grayscale image is smoothed using Gaussian blur to reduce noise.
   - Canny edge detection is applied to identify edges in the image.
3. **Line Detection:**
   - Lines are detected using the Hough Line Transform.
   - Lines are categorized into left and right lanes based on their slopes.
4. **Line Stabilization:**
   - Detected lines are stabilized by averaging their positions over multiple frames.
5. **Visualization:**
   - Steps like grayscale conversion, Gaussian filtering, edge detection, and line drawing are visualized in a combined frame for better clarity.
     
- ##
    Place a video file named `driving.mov` in the project directory.
