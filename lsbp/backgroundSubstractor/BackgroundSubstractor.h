#ifndef BACKGROUNDSUBSTRACTOR_H
#define BACKGROUNDSUBSTRACTOR_H

#include "functionsB.h"


class BackgroundSubstractor{
public:
  BackgroundSubstractor();
  ~BackgroundSubstractor();

  void initialize(string nombre, int h, int w);
  void initialize_model();
  void process_image();
  void process_video(unsigned int);
  void step();
  void scale_svd();
  void scale_lbp();
  void scale_r();
  void scale_t();
  void scale_d();

  void update_R();
  void update_T();
  void update_models();
  void dmin();

  void set_video(Video*);
  void save_show();
  void read_config_file();


  // To manage video
  int h, w;
  Video* current_video;
  VideoWriter* out_video_mask;
  VideoWriter* out_video_real;
  VideoWriter* out_video_svd;
  VideoWriter* out_video_lbp;
  VideoWriter* out_video_grey;

  Mat mask_frame;
  Mat lbp_frame;
  Mat svd_frame;
  Mat grey_frame;
  Mat r_frame;
  Mat t_frame;
  Mat d_frame;


  // cv::namedWindow* mask_window;

  unsigned int batch_pos = 0;
  unsigned int frame_pos = 0;
  // unsigned int batch_size;

  // To manage kernels
  dim3 block;
  dim3 grid;

  // 3d - host
  float* h_B_int;
  int* h_B_lsbp;
  float* h_D;

  //2d - host
  float* h_d;
  float* h_R;
  float* h_T;
  float* h_int;
  float* h_svd;
  int* h_lbp;
  bool* h_mask;

  // 3d - device
  float* d_B_int;
  int* d_B_lsbp;
  float* d_D;

  // 2d - device
  float* d_d;
  float* d_R;
  float* d_T;
  float* d_int;
  float* d_svd;
  int* d_lbp;
  bool* d_mask;

  //model variables
  int s;
  int threshold;
  int HR;
  float R_inc_dec;
  float T_inc;
  float T_dec;
  float R_scale;
  float T_lower;
  float T_upper;
  float R_lower;
  float lbp_threshold;




  // random numbers
  curandState_t* states;

  // Config file
  Config cfg;

  //config options
  bool show_svd;
  bool show_lbp;
  bool show_grey;
  bool show_real;
  bool show_mask;
  bool show_r;
  bool show_d;
  bool show_t;

  bool save_svd;
  bool save_lbp;
  bool save_grey;
  bool save_real;
  bool save_mask;

  string nombre;


};


BackgroundSubstractor::BackgroundSubstractor(){

}

BackgroundSubstractor::~BackgroundSubstractor(){
  delete(h_B_int);
  delete(h_B_lsbp);
  delete(h_D);
  delete(h_d);
  delete(h_R);
  delete(h_T);
  delete(h_svd);

  cudaFree(d_B_int);
  cudaFree(d_B_lsbp);
  cudaFree(d_D);
  cudaFree(d_d);
  cudaFree(d_R);
  cudaFree(d_T);
  cudaFree(d_svd);
  if(save_lbp || show_lbp)
    lbp_frame.release();
  if(save_svd || show_svd)
    svd_frame.release();
  if(save_grey || show_grey)
    grey_frame.release();
  if(show_r)
    r_frame.release();
  if(show_d)
    d_frame.release();
  if(show_t)
    t_frame.release();
}

void BackgroundSubstractor::read_config_file(){
  printf("\n");
  printf("----------Reading configuration file-------------\n");
  printf("\n");

  try{
    cfg.readFile("config.cfg");
  }
  catch(const FileIOException &fioex){
    std::cerr << "I/O error while reading file." << std::endl;
  }
  catch(const ParseException &pex){
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
  }


  try{
    s = cfg.lookup("samples_number");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Number of samples: %d  \n ", s);


  try{
    threshold = cfg.lookup("threshold");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("threshold: %d  \n ", threshold);


  try{
    HR = cfg.lookup("hamming_distance_threshold");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("hamming distance threshold: %d  \n ", HR);

  try{
    lbp_threshold = cfg.lookup("lbp_threshold");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("lbp threshold: %f  \n ", lbp_threshold);


  try{
    R_lower = cfg.lookup("r_lower");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("R_lower: %f  \n ", R_lower);


  try{
    R_inc_dec = cfg.lookup("r_inc_dec");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("R_inc/dec: %f  \n ", R_inc_dec);


  try{
    R_scale = cfg.lookup("R_scale");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("R_scale: %f  \n ", R_scale);


  try{
    T_inc = cfg.lookup("T_inc");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("T_inc: %f  \n ", T_inc);

  try{
    T_dec = cfg.lookup("T_dec");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("T_dec: %f  \n ", T_dec);


  try{
    T_lower = cfg.lookup("T_lower");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("T_lower: %f  \n ", T_lower);


  try{
    T_upper = cfg.lookup("T_upper");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("T_upper: %f  \n ", T_upper);


  try{
    show_lbp = cfg.lookup("show_lbp_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show lbp video: %d  \n ", show_lbp);


  try{
    show_svd = cfg.lookup("show_svd_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show svd video: %d  \n ", show_svd);

  try{
    show_real = cfg.lookup("show_real_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show real video: %d  \n ", show_real);

  try{
    show_mask = cfg.lookup("show_mask_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show mask video: %d  \n ", show_mask);


  try{
    show_grey = cfg.lookup("show_grey_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show grey video: %d  \n ", show_grey);

  try{
    show_r = cfg.lookup("show_r_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show r video: %d  \n ", show_r);

  try{
    show_t = cfg.lookup("show_t_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show t video: %d  \n ", show_t);


  try{
    show_d = cfg.lookup("show_t_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show d video: %d  \n ", show_d);

  try{
    save_lbp = cfg.lookup("save_lbp_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Save lbp video: %d  \n ", save_lbp);


  try{
    save_svd = cfg.lookup("save_svd_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Save svd video: %d  \n ", save_svd);

  try{
    save_real = cfg.lookup("save_real_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Save real video: %d  \n ", save_real);

  try{
    save_mask = cfg.lookup("save_mask_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Save mask video: %d  \n ", save_mask);


  try{
    save_grey = cfg.lookup("save_grey_video");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Save grey video: %d  \n ", save_grey);



  printf("\n");
  printf("----------------------------------------------------\n");
  printf("\n");

}


void BackgroundSubstractor::initialize(string nombre, int h, int w){
  printf("---------Initializing-----------\n");
  printf("\n");



  read_config_file();

  if(show_lbp){
    namedWindow("LSBP", CV_WINDOW_AUTOSIZE);
    moveWindow("LSBP", 600, 50);
  }

  if(show_svd){
    namedWindow("SVD", CV_WINDOW_AUTOSIZE);
    moveWindow("SVD", 600, 350);
  }

  if(show_mask){
    namedWindow("Mascara", CV_WINDOW_AUTOSIZE);
    moveWindow("Mascara", 600, 700);
  }

  if(show_real){
    namedWindow("Real", CV_WINDOW_AUTOSIZE);
    moveWindow("Real", 1200, 50);
  }
  if(show_grey){
    namedWindow("Grey", CV_WINDOW_AUTOSIZE);
    moveWindow("Grey", 1200, 350);
  }

  if(show_d){
    namedWindow("d", CV_WINDOW_AUTOSIZE);
    moveWindow("d", 50, 50);
  }

  if(show_r){
    namedWindow("R", CV_WINDOW_AUTOSIZE);
    moveWindow("R", 50, 350);
  }


  if(show_t){
    namedWindow("T", CV_WINDOW_AUTOSIZE);
    moveWindow("T", 50, 700);
  }


  printf("Height : %d, Width : %d\n", h, w);




  mask_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_lbp || save_lbp)
    lbp_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_svd || save_svd)
    svd_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_grey || save_grey)
    grey_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_d)
    d_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_t)
    t_frame = Mat::zeros(h, w, CV_8UC3);
  if(show_r)
    r_frame = Mat::zeros(h, w, CV_8UC3);


  this->nombre = nombre;
  this->h = h;
  this->w = w;

  block = dim3(16, 16, 1);
  grid = dim3(ceil(h / float(block.x)), ceil(w / float(block.y)));

  printf("Block dimension: %d - %d - %d \n", block.x, block.y, block.z);
  printf("Grid dimension: %d - %d - %d \n", grid.x, grid.y, grid.z);


  printf("Creating host arrays\n");
  // Inicializacion de arrays

  h_B_int = new float[h * w * s];
  h_B_lsbp = new int[h * w * s];
  h_D = new float[h * w * s];

  h_d = new float[h * w];
  h_R = new float[h * w];
  h_T = new float[h * w];

  h_mask = new bool[h * w];
  h_lbp = new int[h * w];
  h_svd = new float[h * w];


  for(int i = 0; i < h; i ++){
    for(int j = 0; j < w; j ++){
      at2d(h_d, i, j, w) = 0.1;
      at2d(h_R, i, j, w) = 18;
      at2d(h_T, i, j, w) = 0.05;

      at2d(h_mask, i, j, w) = false;
      at2d(h_lbp, i, j, w) = 0;
      at2d(h_svd, i, j, w) = 8;

      for(int k = 0; k < s; k ++){
        at3d(h_B_int, i, j, k, w, s) = 1;
        at3d(h_B_lsbp, i, j, k, w, s) = 0;
        at3d(h_D, i, j, k, w, s) = 0.2;
      }
    }
  }

  printf("Creating device arrays\n");

  d_B_int = cuda_array<float>(h * w * s);
  cuda_H2D<float>(h_B_int, d_B_int, h * w * s);

  d_B_lsbp = cuda_array<int>(h * w * s);
  cuda_H2D<int>(h_B_lsbp, d_B_lsbp, h * w * s);

  d_D = cuda_array<float>(h * w * s);
  cuda_H2D<float>(h_D, d_D, h * w * s);


  d_d = cuda_array<float>(h * w);
  cuda_H2D<float>(h_d, d_d, h * w);

  d_R = cuda_array<float>(h * w);
  cuda_H2D<float>(h_R, d_R, h * w);

  d_T = cuda_array<float>(h * w);
  cuda_H2D<float>(h_T, d_T, h * w);

  d_svd = cuda_array<float>(h * w);
  cuda_H2D<float>(h_svd, d_svd, h * w);

  // initialize arrays for svd, lbp, e intensity matrix on __device__
  d_int = cuda_array<float>(h * w);
  d_lbp = cuda_array<int>(h * w);
  d_mask = cuda_array<bool>(h * w);


  printf("Creating curand states\n");
  states = cuda_array<curandState_t>(h * w);


  printf("Initializing curand states\n");
  init_randoms<<<grid, block>>>(time(0), states, h, w);

  cuda_dmin<<<grid, block>>>(d_D, d_d, h, w, s);

  cudaDeviceSynchronize();

  printf("---------End of Initialization-----------\n");

}

void BackgroundSubstractor::set_video(Video* vid){
  current_video = vid;
}

void BackgroundSubstractor::process_image(){

  h_int = current_video->frames_int.at(batch_pos);
  cuda_H2D<float>(h_int, d_int, h * w);

  cuda_svd(d_int, d_svd, h, w, block, grid);
  CHECK(cudaDeviceSynchronize());

  if(show_svd || save_svd){
    cuda_D2H<float>(d_svd, h_svd, h * w);
    scale_svd();
  }

  cuda_lsbp(d_svd, d_lbp, h, w, lbp_threshold, block, grid);
  CHECK(cudaDeviceSynchronize());

  if(show_lbp || save_lbp){
    cuda_D2H<int>(d_lbp, h_lbp, h * w);
    scale_lbp();
  }


  CHECK(cudaDeviceSynchronize());
}

void BackgroundSubstractor::scale_svd(){
  float min = min_v<float>(h_svd, h * w);
  float max = max_v<float>(h_svd, h * w);

  for(int i = 0; i < (h * w); i ++){
    h_svd[i] = int(scaleBetween(h_svd[i], 0, 255, min, max));
  }
}

void BackgroundSubstractor::scale_lbp(){
  int min_lbp = min_v<int>(h_lbp, h * w);
  int max_lbp = max_v<int>(h_lbp, h * w);

  for(int i = 0; i < (h * w); i ++){
    h_lbp[i] = int(scaleBetween(h_lbp[i], 0, 255, min_lbp, max_lbp));
  }
}

void BackgroundSubstractor::scale_r(){
  float min = 0;
  float max = 255;
  for(int i = 0; i < (h * w); i ++){
    h_R[i] = int(scaleBetween(h_R[i], 0, 255, min, max));
  }
}

void BackgroundSubstractor::scale_t(){
  float min = T_lower;
  float max = T_upper;
  for(int i = 0; i < (h * w); i ++){
    h_T[i] = int(scaleBetween(h_T[i], 0, 255, min, max));
  }
}

void BackgroundSubstractor::scale_d(){
  float min = min_v<float>(h_d, h * w);
  float max = max_v<float>(h_d, h * w);
  for(int i = 0; i < (h * w); i ++){
    h_d[i] = int(scaleBetween(h_d[i], 0, 255, min, max));
  }
}

void BackgroundSubstractor::process_video(unsigned int batch_size){
  bool init = true;
  frame_pos = 0;
  batch_pos = 0;
  int batch = 1;
  int c_batch = 0;
  int t_batch = ceil(current_video->num_frames / batch_size);
  while (batch != 0){
    c_batch++;
    batch = ((frame_pos + batch_size) < current_video->num_frames) ? batch_size : (current_video->num_frames - frame_pos);
    if(batch == 0)
      break;
    float p20 = g_20_p(batch);
    int acum_p20 = 0;
    current_video->capture_batch(batch);
    if(init){
      initialize_model();
      init = false;
    }

    batch_pos = 0;
    printf("batch(%d/%d) <<", c_batch, t_batch);
    for(int i = 0 ;i < batch; i++){
      if(acum_p20 >= p20){
        printf("-");
        fflush(stdout);

        acum_p20 = 0;
      }
      step();
      frame_pos++;
      batch_pos++;
      acum_p20++;
    }
    printf(">>\n");
    current_video->erase_frames();
  }

}

void BackgroundSubstractor::initialize_model(){
  printf("--Initializing model-- \n");
  if(save_mask)
    out_video_mask = new VideoWriter("data/video output/" + nombre + "_mask.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));
  if(save_lbp)
    out_video_lbp = new VideoWriter("data/video output/" + nombre + "_lbp.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));
  if(save_svd)
    out_video_svd = new VideoWriter("data/video output/" + nombre + "_svd.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));
  if(save_real)
    out_video_real = new VideoWriter("data/video output/" + nombre + "_real.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));
  if(save_grey)
    out_video_grey = new VideoWriter("data/video output/" + nombre + "_grey.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));

  process_image();
  frame_pos = 0;
  initialization_kernel<<<grid, block>>>(d_B_int, d_B_lsbp, d_int, d_svd, d_lbp, h, w, s, states);
  printf("--end-- \n");
}

void BackgroundSubstractor::step(){
  process_image();

  cuda_step<<<grid, block>>>(d_B_int, d_B_lsbp, d_int, d_svd, d_lbp, d_mask, h, w, s, threshold, HR, d_R);
  cudaDeviceSynchronize();

  if(show_mask || save_mask){
    cuda_D2H(d_mask, h_mask, h * w);
  }


  cuda_update_R<<<grid, block>>>(d_R, d_d, h, w, s, R_scale, R_lower, R_inc_dec);

  cuda_update_T<<<grid, block>>>(d_T, d_mask, d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);


  cudaDeviceSynchronize();

  cuda_update_models<<<grid, block>>>(d_B_int, d_B_lsbp, d_D, d_R, d_T, d_int, d_lbp, d_mask, d_d, h, w, s, states);
  // cudaDeviceSynchronize();
  // cuda_dmin<<<grid, block>>>(d_D, d_d, h, w, s);
  cudaDeviceSynchronize();


  if(show_d){
    cuda_D2H<float>(d_d, h_d, h * w);
    cudaDeviceSynchronize();
    scale_d();
  }

  if(show_t){
    cuda_D2H<float>(d_T, h_T, h * w);
    cudaDeviceSynchronize();
    scale_t();
  }

  if(show_r){
    cuda_D2H<float>(d_R, h_R, h * w);
    cudaDeviceSynchronize();
    scale_r();
  }

  save_show();
}

void BackgroundSubstractor::save_show(){

  Vec3b **ptrLBP;
  Vec3b **ptrSVD;
  Vec3b **ptrMask;
  Vec3b **ptrD;
  Vec3b **ptrR;
  Vec3b **ptrT;
  if (save_lbp || show_lbp){

    ptrLBP = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrLBP[i] = lbp_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(h_lbp, i, j, w);
        ptrLBP[i][j] = Vec3b(d, d, d);
      }
    }
  }
  if (save_svd || show_svd){

    ptrSVD = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrSVD[i] = svd_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(h_svd, i, j, w);
        ptrSVD[i][j] = Vec3b(d, d, d);
      }
    }

  }

  if(show_d){
    ptrD = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrD[i] = d_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(h_d, i, j, w);
        ptrD[i][j] = Vec3b(d, d, d);
      }
    }
  }

  if(show_t){
    ptrT = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrT[i] = t_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(h_T, i, j, w);
        ptrT[i][j] = Vec3b(d, d, d);
      }
    }
  }

  if(show_r){
    ptrR = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrR[i] = r_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(h_R, i, j, w);
        ptrR[i][j] = Vec3b(d, d, d);
      }
    }
  }

  if (save_mask || show_mask){
    ptrMask = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrMask[i] = mask_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        if(at2d(h_mask, i, j, w) == true){
          ptrMask[i][j] = Vec3b(255, 255, 255);
        }
        else{
          ptrMask[i][j] = Vec3b(0, 0, 0);
        }
      }
    }
  }


  if(save_mask)
    out_video_mask->write(mask_frame);
  if(save_lbp)
    out_video_lbp->write(lbp_frame);
  if(save_svd)
    out_video_svd->write(svd_frame);
  if(save_real)
    out_video_real->write(current_video->real_frames.at(batch_pos));
  if(save_grey){
    Vec3b* ptrGrey[h];
    uchar* ptrGrey2[h];
    for(int i = 0; i < h; ++i) {
      ptrGrey[i] = grey_frame.ptr<Vec3b>(i);
      ptrGrey2[i] = current_video->frames.at(batch_pos).ptr<uchar>(i);
      for(int j = 0; j < w; ++j) {
        int d = ptrGrey2[i][j];
        ptrGrey[i][j] = Vec3b(d, d, d);
      }
    }

    out_video_grey->write(grey_frame);
  }
  // imwrite("data/frames output/" + to_string(frame_pos) + ".png", current_video->real_frames.at(batch_pos));
  if(show_svd){
    imshow("SVD", svd_frame);
    delete(ptrSVD);
  }
  if(show_lbp){
    imshow("LSBP", lbp_frame);
    delete(ptrLBP);
  }
  if(show_real)
    imshow("Real", current_video->real_frames.at(batch_pos));
  if(show_grey)
    imshow("Grey", current_video->frames.at(batch_pos));
  if(show_mask)
    imshow("Mascara", mask_frame);

  if(show_r)
    imshow("R", r_frame);
  if(show_d)
    imshow("d", d_frame);
  if(show_t)
    imshow("T", t_frame);

  cv::waitKey(1);
}

#endif
