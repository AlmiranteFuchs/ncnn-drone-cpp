g++ -std=c++17 -g -o demo demo.cpp src/yolo-fastestv2.cpp -I src/include -I include/ncnn lib/libncnn.a `pkg-config --cflags --libs opencv4` -fopenmp

