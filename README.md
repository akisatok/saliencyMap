saliencyMap
===========

Extracting Itti-type saliency maps from a video file or a webcam.
We slso provide a core class for extracting a saliency map from s still image.

Requirements:
* Microsoft Visual C++ / Linux gcc
* OpenCV >= 1.1pre

Usage:
* We prepare two main functions for extracting saliency maps.
* -- FileSMGetSM:  Extracting saliency maps from a video file.
* -- WebcamSMGetSM:  Extracting saliency maps from a webcam.
* The above functions build on a core class named saliencyMap.
* -- The core function: CvMat * saliencyMap::SMGetSM(IplImage *)
* -- Its implementation is quite simple and versatile, works well also with the current version of OpenCV.




