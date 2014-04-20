
saliencyMap::saliencyMap(int height, int width)
{
  // previous frame information
  prev_frame = NULL;
  
  // set Gabor Kernel (9x9)
  GaborKernel0 = cvCreateMat(9, 9, CV_32FC1);
  GaborKernel45 = cvCreateMat(9, 9, CV_32FC1);
  GaborKernel90 = cvCreateMat(9, 9, CV_32FC1);
  GaborKernel135 = cvCreateMat(9, 9, CV_32FC1);
  for(int i=0; i<9; i++) for(int j=0; j<9; j++){
    cvmSet(GaborKernel0, i, j, GaborKernel_0[i][j]); // 0 degree orientation
    cvmSet(GaborKernel45, i, j, GaborKernel_45[i][j]); // 45 degree orientation
    cvmSet(GaborKernel90, i, j, GaborKernel_90[i][j]); // 90 degree orientation
    cvmSet(GaborKernel135, i, j, GaborKernel_135[i][j]); // 135 degree orientation
  }

}
saliencyMap::~saliencyMap(void)
{
  cvReleaseMat(&GaborKernel0);
  cvReleaseMat(&GaborKernel45);
  cvReleaseMat(&GaborKernel90);
  cvReleaseMat(&GaborKernel135);
}

CvMat* saliencyMap::SMGetSM(IplImage *src)
{

  int inputWidth = src->width; // width of the image
  int inputHeight = src->height; // height of the image
  CvSize sSize = cvSize(inputWidth, inputHeight);

  //=========================
  // Intensity and RGB Extraction
  //=========================
  CvMat *R, *G, *B, *I;
  SMExtractRGBI(src, R, G, B, I);

  //=========================
  // Feature Map Extraction
  //=========================
  
  // intensity feature maps
  CvMat* IFM[6];
  IFMGetFM(I, IFM);

  // color feature maps
  CvMat* CFM_RG[6];
  CvMat* CFM_BY[6];
  CFMGetFM(R, G, B, CFM_RG, CFM_BY);

  // orientation feature maps
  CvMat* OFM[24];
  OFMGetFM(I, OFM);

  // motion feature maps
  CvMat* MFM_X[6];
  CvMat* MFM_Y[6];
  MFMGetFM(I, MFM_X, MFM_Y);

  cvReleaseMat(&R);
  cvReleaseMat(&G);
  cvReleaseMat(&B);
  cvReleaseMat(&I);

  //=========================
  // Conspicuity Map Generation
  //=========================

  CvMat *ICM = ICMGetCM(IFM, sSize);
  CvMat *CCM = CCMGetCM(CFM_RG, CFM_BY, sSize);
  CvMat *OCM = OCMGetCM(OFM, sSize);
  CvMat *MCM = MCMGetCM(MFM_X, MFM_Y, sSize);
  for(int i=0; i<6; i++){
    cvReleaseMat(&IFM[i]);
    cvReleaseMat(&CFM_RG[i]);
    cvReleaseMat(&CFM_BY[i]);
    cvReleaseMat(&MFM_X[i]);
    cvReleaseMat(&MFM_Y[i]);
  }
  for(int i=0; i<24; i++) cvReleaseMat(&OFM[i]);

  //=========================
  // Saliency Map Generation
  //=========================

  // Normalize conspicuity maps
  CvMat *ICM_norm, *CCM_norm, *OCM_norm, *MCM_norm;
  ICM_norm = SMNormalization(ICM);
  CCM_norm = SMNormalization(CCM);
  OCM_norm = SMNormalization(OCM);
  MCM_norm = SMNormalization(MCM);
  cvReleaseMat(&ICM);
  cvReleaseMat(&CCM);
  cvReleaseMat(&OCM);
  cvReleaseMat(&MCM);

  // Adding Intensity, Color, Orientation CM to form Saliency Map
  CvMat* SM_Mat = cvCreateMat(sHeight, sWidth, CV_32FC1); // Saliency Map matrix
  float _w_intensity, _w_color, _w_orient, _w_motion;
  smParams.GetWeights(_w_intensity, _w_color, _w_orient, _w_motion);
  cvAddWeighted(ICM_norm, _w_intensity, OCM_norm, _w_orient, 0.0, SM_Mat);
  cvAddWeighted(CCM_norm, _w_color, SM_Mat, 1.00, 0.0, SM_Mat);
  cvAddWeighted(MCM_norm, _w_motion, SM_Mat, 1.00, 0.0, SM_Mat);
  cvReleaseMat(&ICM_norm);
  cvReleaseMat(&CCM_norm);
  cvReleaseMat(&OCM_norm);
  cvReleaseMat(&MCM_norm);

  // Output Result Map
  CvMat* normalizedSM = SMRangeNormalize(SM_Mat);
  CvMat* smoothedSM = cvCreateMat(SM_Mat->height, SM_Mat->width, CV_32FC1); // Saliency Image Output
  cvSmooth(normalizedSM, smoothedSM, CV_GAUSSIAN, 7, 7); // smoothing (if necessary)
  CvMat* SM = cvCreateMat(inputHeight, inputWidth, CV_32FC1); // Saliency Image Output
  cvResize(smoothedSM, SM, CV_INTER_NN);
  cvReleaseMat(&SM_Mat);
  cvReleaseMat(&normalizedSM);
  cvReleaseMat(&smoothedSM);
}

void saliencyMap::SMExtractRGBI(IplImage* inputImage, CvMat* &R, CvMat* &G, CvMat* &B, CvMat* &I)
{
  int height = inputImage->height;
  int width = inputImage->width;

  // convert scale of array elements
  CvMat * src = cvCreateMat(height, width, CV_32FC3);
  cvConvertScale(inputImage, src, 1/256.0);

  // initalize matrix for I,R,G,B
  R = cvCreateMat(height, width, CV_32FC1);
  G = cvCreateMat(height, width, CV_32FC1);
  B = cvCreateMat(height, width, CV_32FC1);
  I = cvCreateMat(height, width, CV_32FC1);

  // split
  cvSplit(src, B, G, R, NULL);

  // extract intensity image
  cvCvtColor(src, I, CV_BGR2GRAY);

  // release
  cvReleaseMat(&src);

}

void SaliencyMap::IFMGetFM(CvMat* src, CvMat* dst[6])
{
  FMGaussianPyrCSD(src, dst);
}

void SaliencyMap::CFMGetFM(CvMat* R, CvMat* G, CvMat* B, CvMat* RGFM[6], CvMat* BYFM[6])
{

  // allocate
  int height = R->height;
  int width = R->width;
  CvMat* tmp1 = cvCreateMat(height, width, CV_32FC1);
  CvMat* tmp2 = cvCreateMat(height, width, CV_32FC1);
  CvMat* RGBMax = cvCreateMat(height, width, CV_32FC1);
  CvMat* RGMin = cvCreateMat(height, width, CV_32FC1);
  CvMat* RGMat = cvCreateMat(height, width, CV_32FC1);
  CvMat* BYMat = cvCreateMat(height, width, CV_32FC1);

  // Max(R,G,B)
  cvMax(R, G, tmp1);
  cvMax(B, tmp1, RGBMax);
  cvMaxS(RGBMax, 0.0001, RGBMax); // to prevent dividing by 0
  // Min(R,G)
  cvMin(R, G, RGMin);

  // R-G
  cvSub(R, G, tmp1);
  // B-Min(R,G)
  cvSub(B, RGMin, tmp2);
  // RG = (R-G)/Max(R,G,B)
  cvDiv(tmp1, RGBMax, RGMat);
  // BY = (B-Min(R,G)/Max(R,G,B)
  cvDiv(tmp2, RGBMax, BYMat);

  // Clamp negative value to 0 for the RG and BY maps
  cvMaxS(RGMat, 0, RGMat);
  cvMaxS(BYMat, 0, BYMat);

  // Obtain [RG,BY] color opponency feature map by generating Gaussian pyramid and performing center-surround difference
  FMGaussianPyrCSD(RGMat, RGFM);
  FMGaussianPyrCSD(BYMat, BYFM);

  // release
  cvReleaseMat(&tmp1);
  cvReleaseMat(&tmp2);
  cvReleaseMat(&RGBMax);
  cvReleaseMat(&RGMin);
  cvReleaseMat(&RGMat);
  cvReleaseMat(&BYMat);
}

void SaliencyMap::OFMGetFM(CvMat* I, CvMat* dst[24])
{
  // Create gaussian pyramid
  CvMat* GaussianI[9];
  FMCreateGaussianPyr(I, GaussianI);

  // Convolution Gabor filter with intensity feature maps to extract orientation feature
  CvMat* tempGaborOutput0[9];
  CvMat* tempGaborOutput45[9];
  CvMat* tempGaborOutput90[9];
  CvMat* tempGaborOutput135[9];
  for(int j=2; j<9; j++){
    int now_height = GaussianI[j]->height;
    int now_width = GaussianI[j]->width;
    tempGaborOutput0[j] = cvCreateMat(now_height, now_width, CV_32FC1);
    tempGaborOutput45[j] = cvCreateMat(now_height, now_width, CV_32FC1);
    tempGaborOutput90[j] = cvCreateMat(now_height, now_width, CV_32FC1);
    tempGaborOutput135[j] = cvCreateMat(now_height, now_width, CV_32FC1);
    cvFilter2D(GaussianI[j], tempGaborOutput0[j], GaborKernel0);
    cvFilter2D(GaussianI[j], tempGaborOutput45[j], GaborKernel45);
    cvFilter2D(GaussianI[j], tempGaborOutput90[j], GaborKernel90);
    cvFilter2D(GaussianI[j], tempGaborOutput135[j], GaborKernel135);
  }
  for(int j=0; j<9; j++) cvReleaseMat(&(GaussianI[j]));

  // calculate center surround difference for each orientation
  CvMat* temp0[6];
  CvMat* temp45[6];
  CvMat* temp90[6];
  CvMat* temp135[6];
  FMCenterSurroundDiff(tempGaborOutput0, temp0);
  FMCenterSurroundDiff(tempGaborOutput45, temp45);
  FMCenterSurroundDiff(tempGaborOutput90, temp90);
  FMCenterSurroundDiff(tempGaborOutput135, temp135);
  for(int i=2; i<9; i++){
    cvReleaseMat(&(tempGaborOutput0[i]));
    cvReleaseMat(&(tempGaborOutput45[i]));
    cvReleaseMat(&(tempGaborOutput90[i]));
    cvReleaseMat(&(tempGaborOutput135[i]));
  }

  // saving the 6 center-surround difference feature map of each angle configuration to the destination pointer
  for(int i=0; i<6; i++){
    dst[i] = temp0[i];
    dst[i+6] = temp45[i];
    dst[i+12] = temp90[i];
    dst[i+18] = temp135[i];
  }
}

void SaliencyMap::MFMGetFM(CvMat* I, CvMat* dst_x[], CvMat* dst_y[])
{
  int height = I->height;
  int width = I->width;

  // convert
  CvMat* I8U = cvCreateMat(height, width, CV_8UC1);
  cvConvertScale(I, I8U, 256);

  // obtain optical flow information
  CvMat* flowx = cvCreateMat(height, width, CV_32FC1);
  CvMat* flowy = cvCreateMat(height, width, CV_32FC1);
  cvSetZero(flowx);
  cvSetZero(flowy);
  if(this->prev_frame!=NULL){
    cvCalcOpticalFlowLK(this->prev_frame, I8U, cvSize(7,7), flowx, flowy);
    cvReleaseMat(&(this->prev_frame));
  }

  // create Gaussian pyramid
  FMGaussianPyrCSD(flowx, dst_x);
  FMGaussianPyrCSD(flowy, dst_y);

  // update
  this->prev_frame = cvCloneMat(I8U);

  // release
  cvReleaseMat(&flowx);
  cvReleaseMat(&flowy);
  cvReleaseMat(&I8U);
}

void FMGaussianPyrCSD(CvMat* src, CvMat* dst[6])
{
  CvMat *GaussianMap[9];
  FMCreateGaussianPyr(src, GaussianMap);
  FMCenterSurroundDiff(GaussianMap, dst);
  for(int i=0; i<9; i++) cvReleaseMat(&(GaussianMap[i]));
}

void FMCreateGaussianPyr(CvMat* src, CvMat* dst[9])
{
  dst[0] = cvCloneMat(src);
  for(int i=1; i<9; i++){
    dst[i] = cvCreateMat(dst[i-1]->height/2, dst[i-1]->width/2, CV_32FC1);
    cvPyrDown(dst[i-1], dst[i], CV_GAUSSIAN_5x5);
  }
}

void FMCenterSurroundDiff(CvMat* GaussianMap[9], CvMat* dst[6])
{
  int i=0;
  for(int s=2; s<5; s++){
    int now_height = GaussianMap[s]->height;
    int now_width = GaussianMap[s]->width;
    CvMat * tmp = cvCreateMat(now_height, now_width, CV_32FC1);
    dst[i] = cvCreateMat(now_height, now_width, CV_32FC1);
    dst[i+1] = cvCreateMat(now_height, now_width, CV_32FC1);
    cvResize(GaussianMap[s+3], tmp, CV_INTER_LINEAR);
    cvAbsDiff(GaussianMap[s], tmp, dst[i]);
    cvResize(GaussianMap[s+4], tmp, CV_INTER_LINEAR);
    cvAbsDiff(GaussianMap[s], tmp, dst[i+1]);
    cvReleaseMat(&tmp);
    i += 2;
  }
}

void SaliencyMap::normalizeFeatureMaps(CvMat *FM[], CvMat *NFM[], int width, int height, int num_maps)
{
  for(int i=0; i<num_maps; i++){
    CvMat * normalizedImage = SMNormalization(FM[i]);
    NFM[i] = cvCreateMat(height, width, CV_32FC1);
    cvResize(normalizedImage, NFM[i], CV_INTER_LINEAR);
    cvReleaseMat(&normalizedImage);
  }
}

CvMat* SaliencyMap::SMNormalization(CvMat* src)
{
  
  CvMat* result = cvCreateMat(src->height, src->width, CV_32FC1);

  // normalize so that the pixel value lies between 0 and 1
  CvMat* tempResult = SMRangeNormalize(src);

  // single-peak emphasis / multi-peak suppression
  double lmaxmean = SMAvgLocalMax(tempResult);
  double normCoeff = (1-lmaxmean)*(1-lmaxmean);
  cvConvertScale(tempResult, result, normCoeff);

  cvReleaseMat(&tempResult);
  return result;
}

CvMat* SaliencyMap::SMRangeNormalize(CvMat* src)
{
  double maxx, minn;
  cvMinMaxLoc(src, &minn, &maxx);
  CvMat* result = cvCreateMat(src->height, src->width, CV_32FC1);
  if(maxx!=minn) cvConvertScale(src, result, 1/(maxx-minn), minn/(minn-maxx));
  else cvConvertScale(src, result, 1, -minn);

  return result;
}

double SMAvgLocalMax(CvMat* src)
{
  int stepsize = DEFAULT_STEP_LOCAL;
  int numlocal = 0;
  double lmaxmean = 0, lmax = 0, dummy = 0;
  CvMat localMatHeader;
  cvInitMatHeader(&localMatHeader, stepsize, stepsize, CV_32FC1, src->data.ptr, src->step);

  for(int y=0; y<src->height-stepsize; y+=stepsize){ // Note: the last several pixels may be ignored.
    for(int x=0; x<src->width-stepsize; x+=stepsize){
      localMatHeader.data.ptr = src->data.ptr+sizeof(float)*x+src->step*y; // get local matrix by pointer trick
      cvMinMaxLoc(&localMatHeader, &dummy, &lmax);
      lmaxmean += lmax;
      numlocal++;
    }
  }

  return lmaxmean/numlocal;
}

CvMat * SaliencyMap::ICMGetCM(CvMat *IFM[], CvSize size)
{
  int num_FMs = 6;

  // Normalize all intensity feature maps
  CvMat * NIFM[6];
  normalizeFeatureMaps(IFM, NIFM, size.width, size.height, num_FMs);

  // Formulate intensity conspicuity map by summing up the normalized intensity feature maps
  CvMat *ICM = cvCreateMat(size.height, size.width, CV_32FC1);
  cvSetZero(ICM);
  for (int i=0; i<num_FMs; i++){
    cvAdd(ICM, NIFM[i], ICM);
    cvReleaseMat(&NIFM[i]);
  }

  return ICM;
}

CvMat * SaliencyMap::CCMGetCM(CvMat *CFM_RG[], CvMat *CFM_BY[], CvSize size)
{
  int num_FMs = 6;
  CvMat* CCM_RG = ICMGetCM(CFM_RG, size);
  CvMat* CCM_BY = ICMGetCM(CFM_BY, size);

  CvMat *CCM = cvCreateMat(size.height, size.width, CV_32FC1);
  cvAdd(CCM_BY, CCM_RG, CCM);

  cvReleaseMat(&CCM_BY);
  cvReleaseMat(&CCM_RG);

  return CCM;
}

CvMat * SaliencyMap::OCMGetCM(CvMat *OFM[], CvSize size)
{
  int num_FMs_perAngle = 6;
  int num_angles = 4;
  int num_FMs = num_FMs_perAngle * num_angles;

  // split feature maps into four sets
  CvMat * OFM0[6];
  CvMat * OFM45[6];
  CvMat * OFM90[6];
  CvMat * OFM135[6];
  for (int i=0; i<num_FMs_perAngle; i++){
    OFM0[i] = OFM[0*num_FMs_perAngle+i];
    OFM45[i] = OFM[1*num_FMs_perAngle+i];
    OFM90[i] = OFM[2*num_FMs_perAngle+i];
    OFM135[i] = OFM[3*num_FMs_perAngle+i];
  }

  // extract conspicuity map for each angle
  CvMat * NOFM_tmp[4];
  NOFM_tmp[0] = ICMGetCM(OFM0, size);
  NOFM_tmp[1] = ICMGetCM(OFM45, size);
  NOFM_tmp[2] = ICMGetCM(OFM90, size);
  NOFM_tmp[3] = ICMGetCM(OFM135, size);

  // Normalize all orientation features map grouped by their orientation angles
  CvMat* NOFM[4];
  for (int i=0; i<4; i++){
    NOFM[i] = SMNormalization(NOFM_tmp[i]);
    cvReleaseMat(&NOFM_tmp[i]);
  }

  // Sum up all orientation feature maps, and form orientation conspicuity map
  CvMat *OCM = cvCreateMat(size.height, size.width, CV_32FC1);
  cvSetZero(OCM);
  for(int i=0; i<4; i++){
    cvAdd(NOFM[i], OCM, OCM);
    cvReleaseMat(&NOFM[i]);
  }

  return OCM;
}

CvMat * SaliencyMap::MCMGetCM(CvMat *MFM_X[], CvMat *MFM_Y[], CvSize size)
{
  return CCMGetCM(MFM_X, MFM_Y, size);
}

CvMat * saliencyMap::SMGetSMFromVideoFrame(CvCapture *input_video, IplImage *&inputFrame_cur, int frameNo, bool retinal_mode)
{

  // read the video's frame size
  CvSize frame_size;
  frame_size.height = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
  frame_size.width = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);

  // get current input frame
  cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, frameNo);
  IplImage * cur_frame = cvQueryFrame(input_video);
  if (cur_frame == NULL){
    printf("Null frame found.");
    exit(1);
  }
  // copy
  inputFrame_cur = cvCloneImage(cur_frame);

  // generate (deterministic) saliency map
  CvMat * SMout = SMGetSM(inputFrame_cur); //itti saliency generation

  // function for retinal filter
  if(retinal_mode){
    bool ib_mode = false;
    CvMat* SMout_withRetinal = IB(SMout, ib_mode);
    cvReleaseMat(&SMout);
    return SMout_withRetinal;
  }
  else return SMout;

}

#ifndef IGNORE_VIDEOINPUT_LIBRARY
CvMat * saliencyMap::SMGetSMFromVideoFrameWebcam(videoInput &vi, int dev_id, IplImage *&inputFrame_cur, bool retinal_mode)
{

  // read the video's frame size
  CvSize frame_size;
  frame_size.height = (int)vi.getHeight(dev_id);
  frame_size.width = (int)vi.getWidth(dev_id);
  std::cerr << " width = " << frame_size.width << " height = " << frame_size.height << std::endl;

  // prepare a buffer for frame data
  static IplImage * prev_inputFrame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  inputFrame_cur = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  cvSetZero(inputFrame_cur);
  char * buffer = inputFrame_cur->imageData;

  // get current input frame
  if(vi.isFrameNew(dev_id)){
    vi.getPixels(dev_id, (unsigned char *)buffer, false, true);
    cvCopy(inputFrame_cur, prev_inputFrame);
  }
  else
    cvCopy(prev_inputFrame, inputFrame_cur);

  // flip input frame (if necessary)
  //cvConvertImage(inputFrame_cur, inputFrame_cur, CV_CVTIMG_FLIP); // maybe does not work well

  // size check and refine
  IplImage * inputFrame_cur2 = NULL;
  float expand_ratio = 1.0;
  CvSize new_size = frame_size;
  int wh_min = MIN(frame_size.width, frame_size.height);
  if(frame_size.width < frame_size.height){ // If the width is smaller than the height, a new width should be set to 256.
    expand_ratio = 256/(float)frame_size.width;
    new_size = cvSize(256, frame_size.height*expand_ratio);
  }
  else {
    expand_ratio = 256/(float)frame_size.height;
    new_size = cvSize(frame_size.width*expand_ratio, 256);
  }
  inputFrame_cur2 = cvCreateImage(new_size, IPL_DEPTH_8U, 3);
  cvResize(inputFrame_cur, inputFrame_cur2, CV_INTER_LINEAR);

  // generate (deterministic) saliency map
  CvMat* SMout2 = SMGetSM(inputFrame_cur2); //itti saliency generation
  cvReleaseImage(&inputFrame_cur2);
  // resize
  CvMat* SMout = cvCreateMat(frame_size.height, frame_size.width, CV_32FC1);
  cvResize(SMout2, SMout, CV_INTER_LINEAR);
  cvReleaseMat(&SMout2);

  // function for retinal filter
  if(retinal_mode){
    bool ib_mode = false;
    CvMat* SMout_withRetinal = IB(SMout, ib_mode);
    cvReleaseMat(&SMout);
    return SMout_withRetinal;
  }
  else return SMout;

}
#endif IGNORE_VIDEOINPUT_LIBRARY
