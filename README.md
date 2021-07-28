# Finger Counter

## Overview

Program for counting fingers on a hand placed in a specified region of interest.

When running the program, at the beginning there should be no objects present in the region of interest (ROI - red rectangle), as moving average is being calculated until 'Wait. getting background' text disappears. If the hand is present in ROI, the results of finger counting will be unstable.

There are two finger counting functions implemented:

* count_fingers - fingers are counted by creating hand's envelope (convex hull) and assuming that fingertips will be farther than 80% of the distance from the centre of the convex hull to farthest point outwards. The method assumes that the hand is in vertical position with fingers pointing upwards.

* count_fingers_2 - fingers are counted by finding defects in convex hull (spots between fingers on the palm of hand). After locating those spots, cosine theorem is used to check if angle between potential fingers is smaller than 90 degrees, if so the spot is couted as being between fingers.

Count_fingers_2 is preferred as it is more stable and presents better results, due to smaller number of assumpions on hand hand orientation.

## Requirements

* Python 3.7
* Opencv 4.5
* Scikit-learn 0.23

## Sources

* [Python for Computer Vision with OpenCV and Deep Learning course](https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/)
* [Hand Detection and Finger Counting Using OpenCV by Madhav Mishra](https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08)
