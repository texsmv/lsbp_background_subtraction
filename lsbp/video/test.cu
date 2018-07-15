#include "Video.h"
#include "../scripts/scripts.h"


int main(){
  string  path = "../../data/videos/video1.mp4";
  video_to_images_resize(path, 100, 150, 100);
  // v.set_size(100,200);

  // v.capture_batch(10);
}
