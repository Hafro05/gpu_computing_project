#include <opencv2/opencv.hpp>

using namespace cv;

void convertImg(cv::Mat img, unsigned char* out_img, int rows, int cols);
void convertImg2(cv::Mat &img, unsigned char* out_img, int rows, int cols);
void CannyCPU(cv::Mat src, cv::Mat &dest, int kernel_size, int L2norm, int low_threshold, int high_threshold);