#include "Video.h"

int main(){
  string  path = "../../data/videos/video1.mp4";
  Video v(path);
  // v.set_size(100,200);
  
  v.capture_batch(10);
}
