main: main.cu
	nvcc main.cu -I/usr/local/include/opencv4 -std=c++14 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lconfig++ -o exe 
