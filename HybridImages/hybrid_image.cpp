#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <math.h>
#include <tuple>

using namespace std;

const string MAIN_WINDOW = "Hybrid Images";


//DEPRECATED
cv::Mat discreteFourierTransform( cv::Mat image ){
    //Perform the discrete fourier transform on an image
    //Expand the image into an optimal size
    cv::Mat paddedImage;
    int rows = cv::getOptimalDFTSize( image.rows );
    int cols = cv::getOptimalDFTSize( image.cols );
    cv::copyMakeBorder( image, paddedImage, 0, rows - image.rows, 0, cols - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );

    //Make place for the complex and real values 
    cv::Mat planes[] = { cv::Mat_<float>(paddedImage), cv::Mat::zeros(paddedImage.size(), CV_32F) };
    cv::Mat complexImage;
    cv::merge( planes, 2, complexImage );

    //make the discrete fourier transform of the image
    cv::dft( complexImage, complexImage );

    //Transform the real and complex values to magnitude 
    cv::split(complexImage, planes);
    cv::magnitude( planes[0], planes[1], planes[0] );
    cv::Mat magnitudeImage = planes[0];

    //Switch to a logarithmic scale so the image is visible 
    magnitudeImage += cv::Scalar::all(1);
    cv::log( magnitudeImage, magnitudeImage );

    //Crop and rearrange the image 
    //magnitudeImage = magnitudeImage(cv::Rect(0,0,magnitudeImage.cols & -2 ))

    return magnitudeImage;
}

double gaussian(int row, int col, double sigma, tuple<int, int> mu){
    double mean = pow(row - std::get<0>(mu), 2.0) + pow(col = std::get<1>(mu), 2.0);
    double coefficient = exp(-1.0 * mean / (2 * pow(sigma,2.0)));
    return coefficient;
}

cv::Mat gaussianKernel(int height, int width, double sigma){
    int center_x = (height % 2 == 1) ? (height/2) + 1 : (height/2);
    int center_y = (width % 2 == 1) ? (width/2) + 1 : (width/2);
    auto mu = std::make_tuple( center_x, center_y );
    cv::Mat kernel = cv::Mat::zeros(height, width, CV_32FC1 );
    for(int row=0; row < kernel.rows; row++ ){
        for(int col=0; col < kernel.cols; col++ ){
            kernel.at<double>(row,col) = gaussian(row, col, sigma, mu);
        }
    }
    int sum = cv::sum(kernel)[0];
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
    for(int col = 0; col < image.cols; col++){ 
        for(int row = 0; row < image.rows; row++){
            int pixel = 0;
            cv::Mat current_section = paddedImage(cv::Rect(col,row,kernel.cols,kernel.rows));
            pixel = cv::sum(current_section.t() * kernel)[0]; 
            result_image.at<double>(col,row) = pixel;
        }
    }
    return result_image;
}

int main(int argc, char** argv){
    string filename("./data/bicycle.bmp");
    cv::Mat _image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat image; 
    _image.convertTo(image, CV_32F);
    cv::namedWindow( MAIN_WINDOW , cv::WINDOW_AUTOSIZE);
    cv::imshow( MAIN_WINDOW, image );
    

    cout << "Creating a gaussian kernel " << endl;
    double sigma = 0.5;
    int size = (int) (8.0f * sigma + 1.0f); 
    if (size % 2 == 0) size++; 
    cv::Mat kernel = gaussianKernel( size,size, sigma );
    cout << "kernel size = " << kernel.size() << endl;
    cout << "Image channels = " << image.channels() << endl;
    cout << "Image size = " << image.size() << endl;

    cv::Mat conv_image;
    if( image.channels() == 1 ){
        conv_image = convolution( image, kernel );
        cv::imshow( MAIN_WINDOW, conv_image );
        cv::waitKey(0);
    }else{
        cv::Mat final_image;
        vector<cv::Mat> channels;
        cv::Mat bgr[image.channels()];
        cv::split( image,bgr );
        for(int channel = 0; channel < image.channels(); channel++){
            conv_image = convolution( bgr[channel], kernel );
            channels.push_back( conv_image );
            // cv::imshow( MAIN_WINDOW,conv_image );
            // cv::waitKey(0);
        }
        cv::merge( channels,final_image );
        cv::imshow( MAIN_WINDOW,final_image );
        cv::waitKey(0);
    }

    
    
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