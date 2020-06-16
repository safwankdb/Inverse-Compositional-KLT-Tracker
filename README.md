# Inverse Compositional KLT Tracker

Python implementation of Kanade-Lucas-Tomasi Tracking Algorithm. This implementation uses the Inverse Compsitional variant as described in [Lucas-Kanade 20 Years On: A Unifying Framework: Part 1](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf).

## Implementation Details

- Uses linear least squares soultion with thresholding to find geometric transform.
- A simple translation is used as the motion model, can be extended easily for affine motion.
- Uses ```cv2.goodFeaturesToTrack()``` for finding keypoints.



## TODO
- [ ] Add video demo.
- [ ] Implement affine motion model.
- [ ] Use RANSAC for robust estimation.
