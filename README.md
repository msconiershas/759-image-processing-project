# 759-image-processing-project

This program’s purpose is to filter images using CUDA and C++. The types of filter options are greyscaling, blurring, and edge detection. The edge detection filter is a Sobel filter. CUDA is a good option for these image processing filters because GPUs work well as a lot of small independent tasks, and that is exactly what image processing is. To use the application, download the file, and run the program with the following format:
 
 <program name> <input image> <option for filtering> 

The options for filtering flags are:

-b or -blur for blurring
-e or -edge for edge detection
-g or -grey or -gray for grayscaling
