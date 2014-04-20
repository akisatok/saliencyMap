#ifndef _SALIENCY_MAP_H_
#define _SALIENCY_MAP_H_

//*************************
// Include Files
//*************************

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <tchar.h>

#include <cv.h>
#include <highgui.h>
#include <videoInput.h>

//*************************
// Constant & Definitions
//*************************

static const float WEIGHT_INTENSITY = 0.30;
static const float WEIGHT_COLOR = 0.30;
static const float WEIGHT_ORIENTATION = 0.20;
static const float WEIGHT_MOTION = 0.20;
static const float RANGEMAX = 255.00;
static const float SCALE_GAUSS_PYRAMID = 1.7782794100389228012254211951927; // = 100^0.125
static const float DEFAULT_STEP_LOCAL = 8;

//*************************
// Class Definition
//*************************

class SMFeatureWeights { // weights for fundamental features
public:
  SMFeatureWeights();
  SMFeatureWeights(float _w_intensity, float _w_color, float _w_orient, float _w_motion);
  ~SMFeatureWeights();
  int SetWeights(float _w_intensity, float _w_color, float _w_orient, float _w_motion);
  int GetWeights(float &_w_intensity, float &_w_color, float &_w_orient, float &_w_motion);
  
protected:
  float w_intensity, w_color, w_orient, w_motion;
};

class SMParams: public SMFeatureWeights { // parameters necessary to construct saliency maps
public:
  SMParams();
  SMParams(float _w_intensity, float _w_color, float _w_orient, float _w_motion, float param, float scale);
  ~SMParams();
  int SetFrameSize(CvSize _frame_size);
  int SetOutputScale(float scale);
  CvSize GetFrameSize();
  float GetRetinalDecayParam();
  float GetOutputScale();

protected:
  CvSize frame_size;
  float outputScale; // resolution parameter of the saliency map (large value = low resolution)
};

class SaliencyMap
{
public:
  // Constructors and descructors
  SaliencyMap(int height, int width);
  ~ParallelSaliencyMap(void);
  // Core: saliency map generation
  CvMat* SMGetSM(IplImage * src);
  // Interfaces
  CvMat* SMGetSMFromVideoFrame(CvCapture * input_video, IplImage * &inputFrame_cur, int frameNo);
  CvMat* SMGetSMFromVideoFrameWebcam(videoInput &vi, int dev_id, IplImage * &inputFrame_cur);
  // Parameters
  SMParams smParams;
  
private:
  CvMat * R, * G, * B, * I;
  CvMat * prev_frame;

  CvMat * GaborKernel0;
  CvMat * GaborKernel45;
  CvMat * GaborKernel90;
  CvMat * GaborKernel135;

private:
  // splitting color channels
  void SMExtractRGBI(IplImage * inputImage, CvMat * &R, CvMat * &G, CvMat * &B, CvMat * &I);
  // extracting feature maps
  void IFMGetFM(CvMat * src, CvMat * dst[6]);
  void CFMGetFM(CvMat * R, CvMat * G, CvMat * B, CvMat * RGFM[6], CvMat * BYFM[6]);
  void OFMGetFM(CvMat * I, CvMat * dst[24]);
  void MFMGetFM(CvMat * I, CvMat * dst_x[6], CvMat * dst_y[6]);
  // normalization
  void normalizeFeatureMaps(CvMat * FM[6], CvMat * NFM[6], int width, int height, int num_maps);
  CvMat* SMNormalization(CvMat * src); // Itti normalization
  CvMat* SMRangeNormalize(CvMat * src); // dynamic range normalization
  // extracting conspicuity maps
  CvMat * ICMGetCM(CvMat *IFM[6], CvSize size);
  CvMat * CCMGetCM(CvMat *CFM_RG[6], CvMat *CFM_BY[6], CvSize size);
  CvMat * OCMGetCM(CvMat *OFM[24], CvSize size);
  CvMat * MCMGetCM(CvMat *MFM_X[6], CvMat *MFM_Y[6], CvSize size);
};

#endif /* _SALIENCY_MAP_H_ */

static const double GaborKernel_0[9][9] = {

{1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06},
{2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05},
{0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076},
{0.00062494, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.00062494},
{0.000921261, 0.006375831, -0.174308068, -0.067914552, 1, -0.067914552, -0.174308068, 0.006375831, 0.000921261},
{0.00062494, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.00062494},
{0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076},
{2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05},
{1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06}

};
static const double GaborKernel_45[9][9] = {

{4.0418E-06, 2.2532E-05, -0.000279806, -0.001028923, 3.79931E-05, 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06},
{2.2532E-05, 0.00092512, 0.002373205, -0.013561362, -0.0229477, 0.000389916, 0.003516954 , 0.000288732, -9.04408E-06},
{-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346 , 0.003516954, 0.000132863},
{-0.001028923, -0.013561362, 0.052928748, 0.46016215, 0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712},
{3.79931E-05, -0.0229477, -0.139178011, 0.249959607, 1, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05},
{0.000744712, 0.000389916, -0.108372072, -0.302454279, 0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923},
{0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806},
{-9.04408E-06, 0.000288732, 0.003516954, 0.000389916, -0.0229477, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05},
{-1.01551E-06, -9.04408E-06, 0.000132863, 0.000744712, 3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06}

};
static const double GaborKernel_90[9][9] = {

{1.85212E-06, 2.80209E-05, 0.000195076, 0.00062494, 0.000921261, 0.00062494, 0.000195076, 2.80209E-05, 1.85212E-06},
{1.28181E-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181E-05},
{-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433},
{-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537},
{0.002010422, 0.030415784, 0.211749204, 0.678352526, 1, 0.678352526, 0.211749204, 0.030415784, 0.002010422},
{-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537},
{-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433},
{1.28181E-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181E-05},
{1.85212E-06, 2.80209E-05, 0.000195076, 0.00062494, 0.000921261, 0.00062494, 0.000195076, 2.80209E-05, 1.85212E-06}

};
static const double GaborKernel_135[9][9] = {

{-1.01551E-06, -9.04408E-06, 0.000132863, 0.000744712, 3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06},
{-9.04408E-06, 0.000288732, 0.003516954, 0.000389916, -0.0229477, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05},
{0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806},
{0.000744712, 0.000389916, -0.108372072, -0.302454279, 0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923},
{3.79931E-05, -0.0229477, -0.139178011, 0.249959607, 1, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05},
{-0.001028923, -0.013561362, 0.052928748, 0.46016215, 0.249959607 , -0.302454279, -0.108372072, 0.000389916, 0.000744712},
{-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863},
{2.2532E-05, 0.00092512, 0.002373205, -0.013561362, -0.0229477, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06},
{4.0418E-06, 2.2532E-05, -0.000279806, -0.001028923, 3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06}

};

