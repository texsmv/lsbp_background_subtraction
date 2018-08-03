#include "lsbp/controller/Controller.h"
#include <cstdlib>

using namespace std;



int main(int argc, char const *argv[]) {
  int height = atoi(argv[1]);
  int width = atoi(argv[2]);
  string nombre_input = argv[3];
  string nombre_output = argv[4];
  int batch_size = atoi(argv[5]);
  int n = atoi(argv[6]);
  Controller c;
  c.init(height, width, nombre_input, nombre_output, batch_size, n);


  return 0;
}
