#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "utils.hpp"
#include "canny_GPU.cuh"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace std::chrono;

int main(int argc, char** argv){
    /*
    List of arguments
    argv[1] = input image
    argv[2] = sobel kernel size
    argv[3] = low threshold
    argv[4] = high threshold
    argv[5] = L2 norm (1: activated; 0:deactivated)
    argv[6] = mode (0: CPU mode; 1: GPU mode; 2: both CPU and GPU)
    */

   // Get the image parameter
   cv::Mat img = imread(argv[1], 0);

   cv::Mat canny_CPU_img = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
   cv::Mat filtered_img = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
   cv::Mat custom_canny_GPU_img =cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
   cv::Mat opencv_canny_GPU_img = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
   cv::Mat diff_img = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));

   // Control the validity of the parameters
   if(img.empty()){
    cerr << "Image not found" << endl;
    exit(1);
   }
   
   if(argc!=7){
    cerr << "The number of parameters is incorrect!" << endl;
    exit(1);
   }

   // Parameters of the algorithm
   unsigned char kernel_size = (unsigned char)atoi(argv[2]);
   int mode = atoi(argv[6]);
   unsigned char L2_norm = (unsigned char)atoi(argv[5]);
   int low_threshold = atoi(argv[3]);
   int high_threshold = atoi(argv[4]);
   double sigma=1.4;

   // Image smoothing
   cv::GaussianBlur(img, filtered_img, cv::Size(3, 3), sigma);

   if(mode==0 || mode==2){
    // Use the CPU implementation of the algorithm (the built-in implementation of openCV)
    auto start = high_resolution_clock::now();

    CannyCPU(filtered_img, canny_CPU_img, kernel_size, L2_norm, low_threshold, high_threshold);
    cv::imwrite("CPU_canny_img.jpg", canny_CPU_img);
    
    auto elapse = std::chrono::system_clock::now() - start;
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    /* Performance computation, results and performance printing ------------ */
    std::cout << " == Performances CPU " << std::endl;
    std::cout << "\t Processing time: " << duration.count() << " (ms)"
                << std::endl;
   }
   if(mode==1 || mode==2){
    // Use our custom GPU implementation of the algorithm
    cudaEvent_t start_gpu, stop_gpu;
    float msecTotal;

    unsigned char *img_h = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    //converts the image passing from a Mat opencv type structure to a dynamically allocated 2D matrix
    convertImg(filtered_img, img_h, img.rows, img.cols);

    unsigned char *out_img_h = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    unsigned char *out_gaussian_img_h = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));

    // Create start event timer
	cudaEventCreate(&start_gpu);
	cudaEventRecord(start_gpu, NULL);

    cannyGPU(img_h, out_img_h, img.rows, img.cols, kernel_size, low_threshold, high_threshold, L2_norm);
    
    // Stop and destroy timer
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(stop_gpu, NULL);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&msecTotal, start_gpu, stop_gpu);

    std::cout << " == Performances GPU" << std::endl;
	std::cout << "\t Processing time: " << msecTotal << " (ms)" << std::endl;

    // Convert the image by passing to a structure of type Mat Opencv
    convertImg2(custom_canny_GPU_img, out_img_h, img.rows, img.cols);
    cv::imwrite("GPU_custom_canny_img.jpg", custom_canny_GPU_img);

    free(img_h);
    free(out_img_h);
   }

   return(EXIT_SUCCESS);
}