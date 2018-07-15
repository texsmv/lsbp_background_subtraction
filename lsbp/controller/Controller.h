#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "functionsC.h"

class Controller{
public:
  Controller();
  ~Controller();
  void init();
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

void Controller::init(){
  VC = new Video("data/videos/video1.mp4");

  BS = new BackgroundSubstractor();
  BS->initialize(500, 500);

  VC->capture_batch(300);
  BS->set_intensities(&(VC->frames_int));
  BS->set_frames(&(VC->frames));


  BS->initialize_model();

  for(int i = 0; i < 300; i++){
    BS->step();
  }


  VC->erase_frames();


}


#endif
