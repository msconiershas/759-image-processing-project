
# Add your own file paths

NVCC=nvcc
OPENCV_LIBPATH=/srv/home/msconiershas/proj/lib64
OPENCV_INCLUDEPATH=/srv/home/msconiershas/proj/include

NVCC_OPTS= `pkg-config --cflags --libs opencv4`
GCC_OPTS= `pkg-config --cflags --libs opencv4`

img_proc: main.o img_load.o blur.o greyscale.o Makefile
	$(NVCC) -o img_proc main.o img_load.o blur.o greyscale.o edge_detect.o  -L $(OPENCV_LIBPATH)  $(NVCC_OPTS)

main.o: main.cu include/img_load.h  include/blur.h 
	$(NVCC) -c main.cu $(NVCC_OPTS)  

img_load.o: img_load.cpp include/img_load.h
	g++  -c img_load.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) 

blur.o: blur.cu include/img_load.h include/blur.h
	$(NVCC) -c blur.cu $(NVCC_OPTS)

greyscale.o: greyscale.cu include/img_load.h include/greyscale.h
	$(NVCC) -c greyscale.cu $(NVCC_OPTS)
	
edge_detect.o: sobel_filter.cu include/img_load.h include/edge_detect.h
	$(NVCC) -c sobel_filter.cu $(NVCC_OPTS)

   

