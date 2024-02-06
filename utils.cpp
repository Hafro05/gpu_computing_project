#include "utils.hpp"

void convertImg2(cv::Mat &img, unsigned char* out_img, int rows, int cols){
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            img.at<uchar>(i, j) = out_img[i*cols+j];
        }
    }

}

void convertImg(cv::Mat img, unsigned char* out_img, int rows, int cols){
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            out_img[i*cols+j] = img.at<uchar>(i, j);
        }
    }
}

void CannyCPU(cv::Mat src, cv::Mat &dest, int kernel_size, int L2norm, int low_threshold, int high_threshold){
    bool L2gradient = false;
    if(L2norm==1) L2gradient = true;

    cv::Canny(src, dest, low_threshold, high_threshold, kernel_size, L2gradient);
}