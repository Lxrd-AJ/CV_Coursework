cmake_minimum_required(VERSION 2.8)

project( HybridImage )

find_package( OpenCV REQUIRED )

add_executable( HybridImage hybrid_image.cpp )

target_compile_options(HybridImage PRIVATE -std=gnu++11)
target_link_libraries( HybridImage ${OpenCV_LIBS} )