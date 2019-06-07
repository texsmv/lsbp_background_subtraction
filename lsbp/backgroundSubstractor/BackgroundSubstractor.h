#ifndef BACKGROUNDSUBSTRACTOR_H
#define BACKGROUNDSUBSTRACTOR_H

#include "functionsB.h"


class Channel{
public:
  Channel(){}
  ~Channel(){}
  Channel(int h, int w, int s){
    h_B_int = new float[h * w * s];
    h_D = new float[h * w * s];

    h_d = new float[h * w];
    h_R = new float[h * w];
    h_T = new float[h * w];

    h_mask = new bool[h * w];


    for(int i = 0; i < h; i ++){
      for(int j = 0; j < w; j ++){
        at2d(h_d, i, j, w) = 0.1;
        at2d(h_R, i, j, w) = 5;
        at2d(h_T, i, j, w) = 10;
        at2d(h_mask, i, j, w) = false;
        for(int k = 0; k < s; k ++){
          at3d(h_B_int, i, j, k, w, s) = 1;
          at3d(h_D, i, j, k, w, s) = 0.2;
        }
      }
    }

    printf("Creating device arrays\n");

    d_B_int = cuda_array<float>(h * w * s);
    cuda_H2D<float>(h_B_int, d_B_int, h * w * s);

    d_D = cuda_array<float>(h * w * s);
    cuda_H2D<float>(h_D, d_D, h * w * s);

    d_d = cuda_array<float>(h * w);
    cuda_H2D<float>(h_d, d_d, h * w);

    d_R = cuda_array<float>(h * w);
    cuda_H2D<float>(h_R, d_R, h * w);

    d_T = cuda_array<float>(h * w);
    cuda_H2D<float>(h_T, d_T, h * w);

    // initialize arrays for svd, lbp, e intensity matrix on __device__
    d_int = cuda_array<float>(h * w);
    d_mask = cuda_array<bool>(h * w);

  }
private:
  int h, w;

  // 3d - host
  float* h_B_int;
  float* h_D;

  //2d - host
  float* h_d;
  float* h_R;
  float* h_T;
  float* h_int;
  bool* h_mask;


  // 3d - device
  float* d_B_int;
  float* d_D;

  // 2d - device
  float* d_d;
  float* d_R;
  float* d_T;
  float* d_int;
  bool* d_mask;

friend class BackgroundSubstractor;
};



class TextureChannel{
public:
  TextureChannel(){}
  ~TextureChannel(){}
  TextureChannel(int h, int w, int s){
    h_B_lsbp = new int[h * w * s];

    h_mask = new bool[h * w];
    h_lbp = new int[h * w];
    h_svd = new float[h * w];


    for(int i = 0; i < h; i ++){
      for(int j = 0; j < w; j ++){

        at2d(h_mask, i, j, w) = false;
        at2d(h_lbp, i, j, w) = 0;
        at2d(h_svd, i, j, w) = 8;

        for(int k = 0; k < s; k ++){
          at3d(h_B_lsbp, i, j, k, w, s) = 0;
        }
      }
    }

    printf("Creating device arrays\n");

    d_B_lsbp = cuda_array<int>(h * w * s);
    cuda_H2D<int>(h_B_lsbp, d_B_lsbp, h * w * s);

    d_svd = cuda_array<float>(h * w);
    cuda_H2D<float>(h_svd, d_svd, h * w);

    // initialize arrays for svd, lbp, e intensity matrix on __device__
    d_lbp = cuda_array<int>(h * w);
    d_mask = cuda_array<bool>(h * w);


  }
private:
  int* h_lbp;
  int* h_B_lsbp;
  float* h_svd;
  bool* h_mask;


  int* d_B_lsbp;
  int* d_lbp;
  float* d_svd;
  bool* d_mask;
friend class BackgroundSubstractor;
};




class BackgroundSubstractor{
public:
  BackgroundSubstractor();
  ~BackgroundSubstractor();

  void initialize(string nombre, int h, int w);
  void initialize_model();
  void process_image();
  void process_video(unsigned int);
  void step();
  void scale_svd(float*);
  void scale_lbp(int*);
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
  void manage_masks();


  // To manage video
  int h, w;
  Video* current_video;
  VideoWriter* out_video_mask;
  VideoWriter* out_video_real;


  Channel* gray;
  Channel* red;
  Channel* green;
  Channel* blue;
  Channel* cb;
  Channel* cr;
  TextureChannel* textura;

  int bool_operator;

  bool* h_global_mask;
  bool* d_global_mask;



  Mat mask_frame;
  Mat texture_frame;


  unsigned int batch_pos = 0;
  unsigned int frame_pos = 0;
  // unsigned int batch_size;

  // To manage kernels
  dim3 block;
  dim3 grid;


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
  bool show_real;
  bool show_mask;
  bool show_texture;

  bool svd;
  bool gray_scale = false;
  bool rgb = false;
  bool texture = false;
  int texture_number;
  bool ycbcr = true;

  bool save_real;
  bool save_mask;

  string nombre;


};


BackgroundSubstractor::BackgroundSubstractor(){

}

BackgroundSubstractor::~BackgroundSubstractor(){



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
    show_texture = cfg.lookup("show_texture");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Show mask video: %d  \n ", show_mask);


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
    svd = cfg.lookup("use_svd");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    gray_scale = cfg.lookup("gray");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    texture = cfg.lookup("texture");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    rgb = cfg.lookup("rgb");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    ycbcr = cfg.lookup("ycbcr");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    texture_number = cfg.lookup("texture_number");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);

  try{
    bool_operator = cfg.lookup("bool_operator");
  }
  catch(const SettingNotFoundException &nfex){
    cerr << "No 'name' setting in configuration file." << endl;
  }
  printf("Use svd: %d  \n ", svd);



  printf("\n");
  printf("----------------------------------------------------\n");
  printf("\n");

}


void BackgroundSubstractor::initialize(string nombre, int h, int w){
  printf("---------Initializing-----------\n");
  printf("\n");



  read_config_file();


  if(show_mask){
    namedWindow("Mascara", CV_WINDOW_AUTOSIZE);
    moveWindow("Mascara", 600, 700);
  }

  if(show_real){
    namedWindow("Real", CV_WINDOW_AUTOSIZE);
    moveWindow("Real", 1200, 50);
  }

  if(show_texture){
    namedWindow("Textura", CV_WINDOW_AUTOSIZE);
    moveWindow("Textura", 150, 100);
  }

  if(show_mask || save_mask){
    mask_frame = Mat::zeros(h, w, CV_8UC3);
  }

  if(show_texture){
    texture_frame = Mat::zeros(h, w, CV_8UC3);
  }

  printf("Height : %d, Width : %d\n", h, w);

  this->nombre = nombre;
  this->h = h;
  this->w = w;


  block = dim3(16, 16, 1);
  grid = dim3(ceil(h / float(block.x)), ceil(w / float(block.y)));

  printf("Block dimension: %d - %d - %d \n", block.x, block.y, block.z);
  printf("Grid dimension: %d - %d - %d \n", grid.x, grid.y, grid.z);
  d_global_mask = cuda_array<bool>(h * w);
  h_global_mask = new bool[h * w];

  if(rgb){
    green = new Channel(h, w, s);
    red = new Channel(h, w, s);
    blue = new Channel(h, w, s);

  }

  if(ycbcr){
    cb = new Channel(h, w, s);
    cr = new Channel(h, w, s);

  }

  if(gray_scale || texture){
    gray = new Channel(h, w, s);
  }
  if(texture){
    textura = new TextureChannel(h, w, s);
  }


  printf("Creating curand states\n");
  states = cuda_array<curandState_t>(h * w);


  printf("Initializing curand states\n");
  init_randoms<<<grid, block>>>(time(0), states, h, w);

  if(gray_scale || texture){
    cuda_dmin<<<grid, block>>>(gray->d_D, gray->d_d, h, w, s);
  }
  if(rgb){
    cuda_dmin<<<grid, block>>>(red->d_D, red->d_d, h, w, s);
    cuda_dmin<<<grid, block>>>(green->d_D, green->d_d, h, w, s);
    cuda_dmin<<<grid, block>>>(blue->d_D, blue->d_d, h, w, s);
  }
  if(ycbcr){
    cuda_dmin<<<grid, block>>>(cb->d_D, cb->d_d, h, w, s);
    cuda_dmin<<<grid, block>>>(cr->d_D, cr->d_d, h, w, s);
  }

  cudaDeviceSynchronize();

  printf("---------End of Initialization-----------\n");

}

void BackgroundSubstractor::set_video(Video* vid){
  current_video = vid;
}

void BackgroundSubstractor::process_image(){
  if(gray_scale || texture){
    gray->h_int = current_video->frames_int.at(batch_pos);
    cuda_H2D<float>(gray->h_int, gray->d_int, h * w);
  }
  if(rgb){
    red->h_int = current_video->frames_red.at(batch_pos);
    cuda_H2D<float>(red->h_int, red->d_int, h * w);

    green->h_int = current_video->frames_green.at(batch_pos);
    cuda_H2D<float>(green->h_int, green->d_int, h * w);

    blue->h_int = current_video->frames_blue.at(batch_pos);
    cuda_H2D<float>(blue->h_int, blue->d_int, h * w);
  }

  if(ycbcr){
    cb->h_int = current_video->frames_cb.at(batch_pos);
    cuda_H2D<float>(cb->h_int, cb->d_int, h * w);

    cr->h_int = current_video->frames_cr.at(batch_pos);
    cuda_H2D<float>(cr->h_int, cr->d_int, h * w);
  }

  if(texture && svd){
    cuda_svd(gray->d_int, textura->d_svd, h, w, block, grid);
  }

  CHECK(cudaDeviceSynchronize());


  if(texture){
    if (svd)
    cuda_texture(textura->d_svd, textura->d_lbp, h, w, lbp_threshold, block, grid, texture_number);
    else
    cuda_texture(gray->d_int, textura->d_lbp, h, w, lbp_threshold, block, grid, texture_number);
  }
  CHECK(cudaDeviceSynchronize());


}

void BackgroundSubstractor::manage_masks(){
  // ----------- getting masks from models -------------------------

  if(gray_scale){
    cuda_mask_one_channel<<<grid, block>>>(gray->d_B_int, gray->d_int, gray->d_mask, h, w, s, threshold, gray->d_R);
  }

  if(texture){
    cuda_mask_texture<<<grid, block>>>(textura->d_B_lsbp, textura->d_lbp, textura->d_mask, h, w, s, threshold, HR);
  }

  if(rgb){
    cuda_mask_one_channel<<<grid, block>>>(green->d_B_int, green->d_int, green->d_mask, h, w, s, threshold, green->d_R);
    cuda_mask_one_channel<<<grid, block>>>(red->d_B_int, red->d_int, red->d_mask, h, w, s, threshold, red->d_R);
    cuda_mask_one_channel<<<grid, block>>>(blue->d_B_int, blue->d_int, blue->d_mask, h, w, s, threshold, blue->d_R);
  }

  if(ycbcr){
    cuda_mask_one_channel<<<grid, block>>>(cb->d_B_int, cb->d_int, cb->d_mask, h, w, s, threshold, cb->d_R);
    cuda_mask_one_channel<<<grid, block>>>(cr->d_B_int, cr->d_int, cr->d_mask, h, w, s, threshold, cr->d_R);
  }

  cudaDeviceSynchronize();
  // ----------------------------------------------------------------

  // ---------------mixing masks------------------------------

  bool b_and = (1 == bool_operator);

  if(gray_scale && texture){
    cudaDeviceSynchronize();
    if(b_and)
      cuda_and_masks_2<<<grid, block>>>(textura->d_mask, gray->d_mask, d_global_mask, h, w);
    else
      cuda_or_masks_2<<<grid, block>>>(textura->d_mask, gray->d_mask, d_global_mask, h, w);
  }
  else if(rgb && texture){
    cuda_or_masks_3<<<grid, block>>>(red->d_mask, green->d_mask, blue->d_mask, d_global_mask, h, w);
    cudaDeviceSynchronize();
    if(b_and)
      cuda_and_masks_2<<<grid, block>>>(textura->d_mask, d_global_mask, d_global_mask, h, w);
    else
      cuda_or_masks_2<<<grid, block>>>(textura->d_mask, d_global_mask, d_global_mask, h, w);
  }
  else if(ycbcr && texture){
    cuda_or_masks_2<<<grid, block>>>(cb->d_mask, cr->d_mask, d_global_mask, h, w);
    cudaDeviceSynchronize();
    if(b_and)
      cuda_and_masks_2<<<grid, block>>>(textura->d_mask, d_global_mask, d_global_mask, h, w);
    else
      cuda_or_masks_2<<<grid, block>>>(textura->d_mask, d_global_mask, d_global_mask, h, w);
  }

  else if(rgb){
    cudaDeviceSynchronize();
    cuda_or_masks_3<<<grid, block>>>(red->d_mask, green->d_mask, blue->d_mask, d_global_mask, h, w);
  }

  else if(ycbcr){
    cudaDeviceSynchronize();
    cuda_or_masks_2<<<grid, block>>>(cb->d_mask, cr->d_mask, d_global_mask, h, w);
  }

  else if(texture){
    cudaDeviceSynchronize();
    cuda_D2D<bool>(textura->d_mask, d_global_mask, h * w);
  }
  else if(gray_scale){
    cudaDeviceSynchronize();
    cuda_D2D<bool>(gray->d_mask, d_global_mask, h * w);
  }

  // ------------------------------------------------------------------------



  cudaDeviceSynchronize();

  if(show_mask || save_mask){

    cuda_D2H<bool>(d_global_mask, h_global_mask, h * w);
    CHECK(cudaDeviceSynchronize());


  }

  if(show_texture){
    cuda_D2H<int>(textura->d_lbp, textura->h_lbp, h * w);
  }



}

void BackgroundSubstractor::scale_svd(float* h_svd){
  float min = min_v<float>(h_svd, h * w);
  float max = max_v<float>(h_svd, h * w);

  for(int i = 0; i < (h * w); i ++){
    h_svd[i] = int(scaleBetween(h_svd[i], 0, 255, min, max));
  }
}

void BackgroundSubstractor::scale_lbp(int* h_lbp){
  int min_lbp = min_v<int>(h_lbp, h * w);
  int max_lbp = max_v<int>(h_lbp, h * w);

  for(int i = 0; i < (h * w); i ++){
    h_lbp[i] = int(scaleBetween(h_lbp[i], 0, 255, min_lbp, max_lbp));
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
  if(save_real)
    out_video_real = new VideoWriter("data/video output/" + nombre + "_real.avi", CV_FOURCC('M','J','P','G'), 30, Size(w, h));

  process_image();
  printf("--!-- \n");
  frame_pos = 0;

  if(texture){
    initialization_kernel_gray_texture<<<grid, block>>>(gray->d_B_int, textura->d_B_lsbp, gray->d_int, textura->d_svd, textura->d_lbp, h, w, s, states);
  }
  else if(gray_scale){
    initialization_kernel_gray<<<grid, block>>>(gray->d_B_int, gray->d_int, h, w, s, states);
  }
  if(rgb){
    initialization_kernel_gray<<<grid, block>>>(red->d_B_int, red->d_int, h, w, s, states);
    initialization_kernel_gray<<<grid, block>>>(green->d_B_int, green->d_int, h, w, s, states);
    initialization_kernel_gray<<<grid, block>>>(blue->d_B_int, blue->d_int, h, w, s, states);
  }
  else if(ycbcr){
    initialization_kernel_gray<<<grid, block>>>(cb->d_B_int, cb->d_int, h, w, s, states);
    initialization_kernel_gray<<<grid, block>>>(cr->d_B_int, cr->d_int, h, w, s, states);

  }
  printf("--end-- \n");
}

void BackgroundSubstractor::step(){

  process_image();

  manage_masks();

  if(gray_scale || texture){
    cuda_update_R<<<grid, block>>>(gray->d_R, gray->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(gray->d_T, gray->d_mask, gray->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);
  }

  if(rgb){
    cuda_update_R<<<grid, block>>>(green->d_R, green->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(green->d_T, green->d_mask, green->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);

    cuda_update_R<<<grid, block>>>(red->d_R, red->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(red->d_T, red->d_mask, red->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);

    cuda_update_R<<<grid, block>>>(blue->d_R, blue->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(blue->d_T, blue->d_mask, blue->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);
  }

  if(ycbcr){
    cuda_update_R<<<grid, block>>>(cb->d_R, cb->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(cb->d_T, cb->d_mask, cb->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);

    cuda_update_R<<<grid, block>>>(cr->d_R, cr->d_d, h, w, s, R_scale, R_lower, R_inc_dec);
    cuda_update_T<<<grid, block>>>(cr->d_T, cr->d_mask, cr->d_d, h, w, s, T_inc, T_dec, T_lower, T_upper);
  }


  cudaDeviceSynchronize();

  if(rgb){
    cuda_update_models_gray<<<grid, block>>>(green->d_B_int, green->d_D, green->d_R, green->d_T, green->d_int, green->d_mask, green->d_d, h, w, s, states);
    cuda_update_models_gray<<<grid, block>>>(red->d_B_int, red->d_D, red->d_R, red->d_T, red->d_int, red->d_mask, red->d_d, h, w, s, states);
    cuda_update_models_gray<<<grid, block>>>(blue->d_B_int, blue->d_D, blue->d_R, blue->d_T, blue->d_int, blue->d_mask, blue->d_d, h, w, s, states);
  }

  if(ycbcr){
    cuda_update_models_gray<<<grid, block>>>(cb->d_B_int, cb->d_D, cb->d_R, cb->d_T, cb->d_int, cb->d_mask, cb->d_d, h, w, s, states);
    cuda_update_models_gray<<<grid, block>>>(cr->d_B_int, cr->d_D, cr->d_R, cr->d_T, cr->d_int, cr->d_mask, cr->d_d, h, w, s, states);
  }

  if(gray_scale){
    cuda_update_models_gray<<<grid, block>>>(gray->d_B_int, gray->d_D, gray->d_R, gray->d_T, gray->d_int, gray->d_mask, gray->d_d, h, w, s, states);
  }
  else if(texture){
    cuda_update_models_gray_texture<<<grid, block>>>(gray->d_B_int, textura->d_B_lsbp, gray->d_D, gray->d_R, gray->d_T, gray->d_int, textura->d_lbp, gray->d_mask, gray->d_d, h, w, s, states);
  }
  // cudaDeviceSynchronize();
  // cuda_dmin<<<grid, block>>>(d_D, d_d, h, w, s);
  cudaDeviceSynchronize();



  save_show();
}

void BackgroundSubstractor::save_show(){


  Vec3b **ptrMask;
  Vec3b **ptrText;

  if (save_mask || show_mask){
    ptrMask = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrMask[i] = mask_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {


        if(at2d(h_global_mask, i, j, w) == true){
          ptrMask[i][j] = Vec3b(255, 255, 255);
        }
        else{
          ptrMask[i][j] = Vec3b(0, 0, 0);
        }

      }
    }
  }

  if (show_texture){

    ptrText = new Vec3b*[h];
    for(int i = 0; i < h; ++i) {
      ptrText[i] = texture_frame.ptr<Vec3b>(i);
      for(int j = 0; j < w; ++j) {
        int d = at2d(textura->h_lbp, i, j, w);
        ptrText[i][j] = Vec3b(d, d, d);
      }
    }
  }




  if(save_mask)
    out_video_mask->write(mask_frame);
  if(save_real)
    out_video_real->write(current_video->real_frames.at(batch_pos));
  // imwrite("data/frames output/" + to_string(frame_pos) + ".png", current_video->real_frames.at(batch_pos));

  if(show_real){
    imshow("Real", current_video->real_frames.at(batch_pos));
  }
  if(show_mask){
    imshow("Mascara", mask_frame);
  }
  if(show_texture){
    imshow("Textura", texture_frame);
  }

  float ratio = calc_ratio(h_global_mask, h, w);
  cout<<"Ratio : "<<ratio<<endl;
  if(ratio >= 0.001){
    cout<<"ALARMA!!!"<<endl;
  }


  cv::waitKey(1);
}

#endif
