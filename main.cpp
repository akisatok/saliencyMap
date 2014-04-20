
//----------------------------------------------------------------------------- 
// Function: WebcamSMGetSM
//-----------------------------------------------------------------------------
void webcamSSMSaliencyCore(videoInput &vi, int dev_id, saliencyMap *SM, // input
                           IplImage *& inputFrame, IplImage *& SMout)  // output
{
  // generate saliency map from the designated frame of the input video
  bool retinal_mode = true;
  CvMat * SMout = SM->SMGetSMFromVideoFrameWebcam(vi, dev_id, inputFrame, retinal_mode);
}
void WebcamSSMSaliency()
{
  // create a videoInput object
  videoInput vi;
  // count the number of devices available
  int num_devs = vi.listDevices();
  // short cut (currently, we assume 2 devices at most)
  int dev0 = 0;
  //int dev1 = 1;
  // setup the devices
  vi.setupDevice(dev0, DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT);
  //vi.setupDevice(dev1, 640, 480);

  // Read the video's frame size 
  CvSize frame_size;
  frame_size.width  =	(int)vi.getWidth (dev0);
  frame_size.height =	(int)vi.getHeight(dev0);

  // Create windows for visualizing the output.
  char * window_name_input      = "Input Video";
  char * window_name_saliency   = "Saliency Video";
  char * window_name_stochastic = "Eye Focusing Density Video";
  cvNamedWindow(window_name_input,      CV_WINDOW_AUTOSIZE);
  cvNamedWindow(window_name_saliency,   CV_WINDOW_AUTOSIZE);
  cvNamedWindow(window_name_stochastic, CV_WINDOW_AUTOSIZE); 

  // deterministic and stochastic saliency maps
  saliencyMap * SM  = new saliencyMap  (frame_size.height, frame_size.width);
  // create retinal filter mask
  SM->createRetinalFilterMask(frame_size);

  // loop
  while(1){
    // attention estimation
    IplImage * inputFrame = NULL;
    IplImage * outputSM = NULL;
    webcamSSMSaliencyCore(vi, dev0, SM, inputFrame, outputSM);

    // show input and output on display
    cvShowImage(window_name_input,      inputFrame);
    cvShowImage(window_name_saliency,   outputSM );
    // wait for key pressing
    int key = cvWaitKey(10);   // Please see the key table: http://homepage1.nifty.com/kabayan/java2/data01/apt053.html
    if(key==27){  // ESC --> abort
      break;
    }
    else if(key==32){  // SPACE --> reset
      // delete
      delete SM;
      // deterministic and stochastic saliency maps
      saliencyMap * SM  = new saliencyMap  (frame_size.height, frame_size.width);
      // create retinal filter mask
      SM->createRetinalFilterMask(frame_size);
    }

    // memory cleanup
    cvReleaseImage(&inputFrame);
    cvReleaseImage(&outputSM);

  }

  // release
  vi.stopDevice(dev0);
  //	vi.stopDevice(dev1);
  delete SM;

  return;
}

void FileSMGetSM(char * input_video_name, char * output_video_name, char * output_data_name)
{

  // open the input video
  CvCapture* input_video = cvCaptureFromAVI(input_video_name);
  if(input_video == NULL) myError("Could not load video file");

  // read the video's frame size
  CvSize frame_size;
  frame_size.height = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
  frame_size.width = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);
  //// determine the number of frames in the AVI.
  cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_AVI_RATIO, 1. );
  long number_of_frames = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES);
  // get FPS of input videop
  int fps = (int)cvGetCaptureProperty(input_video, CV_CAP_PROP_FPS);
  // return to the beginning
  cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, 0);

  // initalize video writer setting
  CvVideoWriter * output_video_writer = cvCreateVideoWriter(output_video_name, CV_FOURCC('I','4','2','0') /* raw video */, fps, frame_size, 1);

  // create windows for visualizing the output.
  char * window_name_input = "Input Video";
  char * window_name_saliency = "Saliency Video";
  cvNamedWindow(window_name_input, CV_WINDOW_AUTOSIZE);
  cvNamedWindow(window_name_saliency, CV_WINDOW_AUTOSIZE);

  // initalize starting frame no. for extraction
  long frameNo = 0;
  long start_frameNo = 10;

  // saliency maps
  saliencyMap * SM = new saliencyMap(frame_size.height, frame_size.width);

  // create retinal filter mask
  SM->createRetinalFilterMask(frame_size);

  long totalTime = 0;

  // open the file for log output
  ofstream logfile;
  logfile.open(output_data_name);

  // insert dummy frames
  IplImage * frame_for_write = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  cvSetZero(frame_for_write);
  for(frameNo=0; frameNo<start_frameNo; frameNo++){
    // write dummy video frames
    cvWriteFrame(output_video_writer, frame_for_write);
  }

  // loop (frames)
  for(frameNo=start_frameNo; frameNo<number_of_frames; frameNo++){
    // begin time tracking clock
    timeBeginPeriod(1);
    t1 = timeGetTime();

    // generate saliency map from the designated frame of the input video
    bool retinal_mode = true;
    IplImage *inputFrame = NULL;
    CvMat * SMout = SM->SMGetSMFromVideoFrame(input_video, inputFrame, frameNo, retinal_mode);
    t2 = timeGetTime();
    logfile << "Get a saliency map... " << (t2-t1) << " ms" << endl;

    // print the score and time to the console
    t2 = timeGetTime();
    logfile << "Completed saliecny extraction for frame#" << frameNo;
    logfile << "..." << (t2-t1) << "ms" << endl << endl;
    totalTime += t2-t1;
    timeEndPeriod(1);

    // show input and output on display
    cvShowImage(window_name_input, inputFrame);
    cvShowImage(window_name_saliency, SMout );
    cvWaitKey(5);

    // write output video frames
    IplImage tmp_header;
    IplImage *outputSM_8U = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    cvConvertScale(cvGetImage(SMout, &tmp_header), outputSM_8U, 256);
    IplImage * outputSM_8U_flip = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    cvConvertImage(outputSM_8U, outputSM_8U_flip, CV_CVTIMG_FLIP); // I am not sure why we have to flip it
    cvCvtColor(outputSM_8U_flip, frame_for_write, CV_GRAY2BGR);
    cvWriteFrame(output_video_writer, frame_for_write);

    // memory cleanup
    cvReleaseImage(&inputFrame);
    cvReleaseMat(&SMout);
    cvReleaseImage(&outputSM_8U);
    cvReleaseImage(&outputSM_8U_flip);

    logfile.flush();
  }

  logfile << "\n FINAL: " << "\t avg FPS: " << (number_of_frames-START_FRAME_INDENT)*1000/(double)totalTime << endl;

  // release input capture and output files
  cvReleaseImage(&frame_for_write);
  cvReleaseCapture(&input_video);
  cvReleaseVideoWriter(&output_video_writer);

  delete SM;

  return;
}


int main( int argc, char* argv[] ){

  // mode selection
  char * APPLICATION_MODE = argv[1];

  if(strcmp(APPLICATION_MODE, "WEBCAM_MODE")==0){ // webcam input
    WebcamSMGetSM();
  }
  else if(strcmp(APPLICATION_MODE,"FILE_MODE")==0){ // video file input
    char * input_video_name = argv[2];
    char * output_video_name = argv[3];
    char * output_data_name = argv[4];
    FileSMGetSM(input_video_name, output_video_name, output_data_name);
  }
  else {
    perror("Mode error.");
  }

  return 0;

}

