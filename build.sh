# g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o hybrid_image  hybrid_image.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect  -lopencv_stitching -lopencv_imgcodecs -std=c++11

g++ -std=gnu++11 hybrid_image.cpp -o hybrid_image `pkg-config --libs opencv`