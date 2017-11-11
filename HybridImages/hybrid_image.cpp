#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <math.h>
#include <tuple>

using namespace std;

const string MAIN_WINDOW = "Hybrid Image";
const string LOW_WINDOW = "Low Frequency Window";
const string HIGH_WINDOW = "High Frequency Window";

cv::Mat read_image(string filename){
    cv::Mat _image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat image;
    _image.convertTo(image, CV_32F);
    return image;
}

double gaussian(int row, int col, double sigma, tuple<int, int> mu){
    double mean = pow(row - std::get<0>(mu), 2) + pow(col - std::get<1>(mu), 2);
    double coefficient = exp((-1 * mean) / (2 * pow(sigma,2)));
    // double top = exp(-((pow(row,2.0) + pow(col,2.0)) / 2 * pow(sigma,2.0)));
    // double bottom = 2 * M_PI * pow(sigma,2.0);
    // double coefficient = top / bottom;
    return coefficient;
}

/**
 * TODO: Check for bug in gaussian kernel function
*/
cv::Mat gaussianKernel(int height, int width, double sigma){
    int center_x = (height % 2 == 1) ? (height/2) + 1 : (height/2);
    int center_y = (width % 2 == 1) ? (width/2) + 1 : (width/2);
    auto mu = std::make_tuple( center_x, center_y );
    cout << "Kernel mean = (" << center_x << "," << center_y << ")" << endl;
    cv::Mat kernel = cv::Mat::zeros(height, width, CV_32FC1 );
    for(int row=1; row <= kernel.rows; row++ ){
        for(int col=1; col <= kernel.cols; col++ ){
            kernel.at<double>(col-1,row-1) = gaussian(col, row, sigma, mu);
        }
    }
    int sum = cv::sum(kernel)[0];
    kernel = kernel / sum;
    return kernel;
}

cv::Mat gaussian_template( int win_size, double sigma ){
    int centre = floor(win_size/2) + 1;
    double sum = 0;
    cout << "Center = " << centre << endl;
    cv::Mat kernel = cv::Mat::zeros(win_size, win_size, CV_32FC1 );
    for(int i=1; i <= win_size; i++){
        for(int j=1; j <= win_size; j++){
            int top = pow((j - centre), 2) + pow((i - centre), 2);
            cout << top << "\t";
            kernel.at<double>(j,i) = exp(-top / (2 * sigma * sigma));
            sum = sum + kernel.at<double>(j,i);
        }
    }

    kernel = kernel / sum;
    return kernel;
}

/**
 * Convolves the image and kernel using a discrete fourier transform
 * Works on a single channel image, use convolution3 for 3 channel images
*/
cv::Mat convolution(cv::Mat image, cv::Mat kernel){
    //Pad the image 
    int height = kernel.rows;
    int width = kernel.cols;
    int center_x = (height % 2 == 1) ? (height/2) + 1 : (height/2);
    int center_y = (width % 2 == 1) ? (width/2) + 1 : (width/2);
    int top_pad = kernel.rows - center_x;
    int side_pad = kernel.cols - center_y;
    cv::Mat paddedImage;
    cv::copyMakeBorder( image, paddedImage, top_pad, top_pad, side_pad, side_pad, cv::BORDER_CONSTANT, 0);
    cv::Mat result_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1 );;

    //begin image convolution
    for(int row = 0; row < image.rows; row++){ 
        for(int col = 0; col < image.cols; col++){
            int pixel = 0;
            cv::Mat current_section = paddedImage(cv::Rect(col,row,kernel.cols,kernel.rows));
            pixel = cv::sum(current_section * kernel.t())[0]; 
            result_image.at<double>(row,col) = pixel;
        }
    }
    return result_image;
}

cv::Mat convolution_channels(cv::Mat image, cv::Mat kernel){
    cv::Mat conv_image;
    cv::Mat final_image;
    vector<cv::Mat> channels;
    cv::Mat bgr[image.channels()];
    cv::split( image,bgr );
    for(int channel = 0; channel < image.channels(); channel++){
        conv_image = convolution( bgr[channel], kernel );
        channels.push_back( conv_image );
    }
    cv::merge( channels,final_image );
    return final_image;
}

cv::Mat hybrid_image(cv::Mat image1, cv::Mat image2, cv::Mat kernel){
    cv::Mat low_img1 = convolution_channels( image1, kernel );
    cv::imshow( LOW_WINDOW, low_img1 );
    cv::Mat _img2 = convolution_channels( image2, kernel );
    cv::Mat high_img2;
    cv::Mat result;
    cv::subtract(image2, _img2, high_img2, cv::Mat(), CV_32F);
    cv::imshow( HIGH_WINDOW, high_img2 );
    cv::add( low_img1, high_img2, result, cv::Mat(), CV_32F );
    return result;
}

int main(int argc, char** argv){
    string filename("./data/dog.bmp");
    string filename2("./data/cat.bmp");

    cv::Mat image = read_image(filename);
    cv::Mat image2 = read_image(filename2);
    
    cv::namedWindow( MAIN_WINDOW , cv::WINDOW_AUTOSIZE);
    cv::namedWindow( LOW_WINDOW, cv::WINDOW_AUTOSIZE );
    cv::namedWindow( HIGH_WINDOW, cv::WINDOW_AUTOSIZE );
    
    cout << "Creating a gaussian kernel " << endl;
    double sigma = 0.3;
    int size = (int) (8.0f * sigma + 1.0f); 
    if (size % 2 == 0) size++; 
    cv::Mat kernel = gaussianKernel( size, size, sigma );
    // cv::Mat kernel = gaussian_template( size, sigma );
    cout << "kernel size = " << kernel.size() << endl;
    cout << "Gaussian Kernel\n" << kernel << endl;
    cout << "Image channels = " << image.channels() << endl;
    cout << "Image2 channels = " << image2.channels() << endl;
    cout << "Image size = " << image.size() << endl;
    cout << "Image2 size = " << image2.size() << endl;

    cv::Mat hybrid = hybrid_image( image, image2, kernel );
    cout << "Hybrid Image of " << filename << " and " << filename2 << endl;
    cv::imshow( MAIN_WINDOW, hybrid );
    cv::waitKey(0);

    cout << "End of program ..." << endl;
    exit(0);
}

/**
Resources 
- Main = https://jeremykun.com/2014/09/29/hybrid-images/
- 2nd Main - https://github.com/cornellcs56702017/Cornell-CS5670-2017/blob/master/Project1_Hybrid_Images/hybrid.py and http://www.cs.cornell.edu/courses/cs5670/2017sp/projects/pa1/index.html
- http://cs.brown.edu/courses/cs143/2013/proj1/
- https://gist.github.com/omaflak/aca9d0dc8d583ff5a5dc16ca5cdda86a
- http://hipersayanx.blogspot.co.uk/2015/06/image-convolution.html
- Image kernel visualisation - http://setosa.io/ev/image-kernels/
*/