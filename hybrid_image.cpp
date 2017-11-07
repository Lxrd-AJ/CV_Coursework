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

double gaussian(int row, int col, double sigma, tuple<int, int> mu, bool highPass = false){
    double mean = pow(row - std::get<0>(mu), 2.0) + pow(col = std::get<1>(mu), 2.0);
    double coefficient = exp(-1.0 * mean / (2 * pow(sigma,2.0)));
    return (highPass ? (1 - coefficient) : coefficient);
}

cv::Mat gaussianKernel(int height, int width, double sigma, bool highPass = false){
    int center_x = (height % 2 == 1) ? (height/2) + 1 : (height/2);
    int center_y = (width % 2 == 1) ? (width/2) + 1 : (width/2);
    auto mu = std::make_tuple( center_x, center_y );
    cv::Mat kernel = cv::Mat::zeros(height, width, CV_8UC1 );
    for(int row=0; row < kernel.rows; row++ ){
        for(int col=0; col < kernel.cols; col++ ){
            kernel.at<double>(row,col) = gaussian(row, col, sigma, mu);
        }
    }

    return kernel;
}

// cv::Mat extractRegion(cv::Mat image, int start, int stop, int width, int height){
    
// }

/**
 * Convolves the image and kernel using a discrete fourier transform
 * Works on a single channel image, use convolution3 for 3 channel images
*/
cv::Mat convolution(cv::Mat image, cv::Mat kernel){
    cout << image.size() << endl;
    //Pad the image 
    int height = kernel.rows;
    int width = kernel.cols;
    int center_x = (height % 2 == 1) ? (height/2) + 1 : (height/2);
    int center_y = (width % 2 == 1) ? (width/2) + 1 : (width/2);
    int top_pad = kernel.rows - center_x;
    int side_pad = kernel.cols - center_y;
    cv::Mat paddedImage;

    cout << "Image size (" << image.rows << "," << image.cols << ")" << endl;
    cout << "Kernel size (" << height << "," << width << ")" << endl;

    cv::copyMakeBorder( image, paddedImage, top_pad, top_pad, 1, 1, cv::BORDER_CONSTANT, 0);
    cout << paddedImage.row(2) << endl;
    cv::Mat result_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1 );;

    //begin image convolution
    for(int col = 0; col < image.cols; col++){ 
        for(int row = 0; row < image.rows; row++){
            int pixel = 0;

            cv::Mat current_section = paddedImage(cv::Rect(col,row,kernel.cols,kernel.rows));
            cout << "Current section" << endl;
            cout << current_section << endl;
            pixel = cv::sum(current_section * kernel)[0]; //TODO: Fix bug in matrix multiplication

            // for(int j = 0; j < kernel.cols; j++){
            //     for(int i=0; i < kernel.rows; i++){
            //         int y = (row + i) - center_y;
            //         int x = (col + j) - center_x;
            //         cout << "x = " << x << " , y = " << y << endl;
            //         pixel += paddedImage.at<double>(x,y) * kernel.at<double>(i,j);
            //     }
            // }

            result_image.at<double>(col,row) = pixel;
        }
    }
    return result_image;
}

//Works on a 3 channel image.
cv::Mat convolution3(cv::Mat image, cv::Mat kernel){
    //Assume it has 3 channels
    vector<cv::Mat> channels(3);
    cv::split( image, channels );
    cv::Mat blueChannel = channels[0];
    cv::Mat greenChannel = channels[1];
    cv::Mat redChannel = channels[2];
    //TODO: Operate on the channels
    //Merge the channels back together
    
    //TODO: Fix = cv::merge( channels, 3, image );

    return image; //TODO
}

int main(int argc, char** argv){
    string filename("./data/marilyn.bmp");
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::namedWindow( MAIN_WINDOW , cv::WINDOW_AUTOSIZE);
    cv::imshow( MAIN_WINDOW, image );
    //cv::waitKey(0);

    //Print out many gaussians
    int count = 5;
    auto mu = std::make_tuple(3,5);
    // for(int row = 0; row < count; row++){
    //     for(int col = 0; col < count; col++){
    //         auto gauss = gaussian(row, col, 3.0, mu);
    //         cout << "Gaussian at (" << row << "," << col << ") = " << gauss << endl;
    //     }
    // }

    cout << "Creating a gaussian kernel " << endl;
    cv::Mat kernel = gaussianKernel( 5,7, 0.9 );
    cout << kernel << endl;
    cout << "***Here ****" << endl;

    cout << "Image channels = " << image.channels() << endl;

    cv::Mat conv_image = convolution( image, kernel );
    cv::imshow( MAIN_WINDOW, conv_image );
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