#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "functionsC.h"

class Controller{
public:
  Controller();
  ~Controller();
  void init(unsigned int, unsigned int, string, string, unsigned int, unsigned int);
private:
  BackgroundSubstractor* BS;
  Video* VC;

};


Controller::Controller(){

}

Controller::~Controller(){
  delete(VC);
  delete(BS);
}

void Controller::init(unsigned int height, unsigned int width, string nombre_input, string nombre_output, unsigned int batch_size, unsigned int n){
  VC = new Video("data/video_input/" + nombre_input);
  std::cout << "Video: " << "data/video_input/" + nombre_input<< std::endl;

  VC->set_size(height, width);

  if(n != 0){
    VC->num_frames = n;
  }
  printf("%d\n", VC->num_frames);

  BS = new BackgroundSubstractor();
  BS->initialize(nombre_output, height, width);
  
  BS->set_video(VC);
  BS->process_video(batch_size);

}


#endif
