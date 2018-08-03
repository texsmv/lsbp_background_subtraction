main: main.cu
	nvcc main.cu -std=c++11 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lconfig++ -o exe 
