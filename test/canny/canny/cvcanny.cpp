
#include "cv.h"
#include "highgui.h"

char wndname[] = "Edge";  
char tbarname[] = "Threshold"; 
int edge_thresh = 1;
IplImage *image = 0, *cedge = 0, *gray = 0, *edge = 0; 
// 定义跟踪条的callback 函数
 
void on_trackbar(int h) 
{  
    cvSmooth( gray, edge, CV_BLUR, 3, 3, 0 ); 
    cvNot( gray, edge );  
    // 对灰度图像进行边缘检测
    cvCanny(gray, edge, (float)edge_thresh, (float)edge_thresh*3, 3); 
    cvZero( cedge ); 
    // copy edge points  
    cvCopy( image, cedge, edge ); 
    // 显示图 
    cvShowImage(wndname, cedge); 
}  
int main( int argc, char** argv ) 
{  
    char* filename = argc == 2 ? argv[1] : (char*)"emosue.jpg";      
    if( (image = cvLoadImage( filename, 1)) == 0 ) 
        return -1;  
    // Create the output image
    cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);  
    // 将彩色图像转换为灰度图像
    gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);  
    edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);  
    cvCvtColor(image, gray, CV_BGR2GRAY); 
    // Create a window  
    cvNamedWindow(wndname, 1);
    // create a toolbar   
    cvCreateTrackbar(tbarname, wndname, &edge_thresh, 100, on_trackbar); 
    // Show the image 
    on_trackbar(1);
	    // Wait for a key stroke; the same function arranges events processing 
    cvWaitKey(0);  
    cvReleaseImage(&image); 
    cvReleaseImage(&gray); 
    cvReleaseImage(&edge); 
    cvDestroyWindow(wndname); 
    return 0; 
} 
/*******代码中函数说明
1、cvSmooth，其函数声明为：
cvSmooth( const void* srcarr, void* dstarr, int smoothtype,int param1, int param2, double param3 )  
cvSmooth函数的作用是对图象做各种方法的图象平滑。其中，srcarr为输入图象；dstarr为输出图象；param1为平滑操作的第一个参数；
param2为平滑操作的第二个参数（如果param2值为0，则表示它被设为param1）；param3是对应高斯参数的标准差。参数smoothtype是图象
平滑的方法选择，主要的平滑方法有以下五种：CV_BLUR_NO_SCALE：简单不带尺度变换的模糊，即对每个象素在param1×param2领域求和。
CV_BLUR：对每个象素在param1×param2邻域求和并做尺度变换1/（param1?param2）。CV_GAUSSIAN：对图像进行核大小为param1×param2
的高斯卷积。CV_MEDIAN：对图像进行核大小为param1×param1 的中值滤波（邻域必须是方的）。CV_BILATERAL：双向滤波，应用双向 3x3 
滤波，彩色设置为param1，空间设置为param2。
2、void cvNot(const CvArr* src,CvArr* dst);  
函数cvNot()会将src中的每一个元素的每一位取反，然后把结果赋给dst。因此，一个值为0x00的8位图像将被映射到0xff，而值为0x83的图像
将被映射到0x7c。
3、void cvCanny( const CvArr* image, CvArr* edges, double threshold1,double threshold2, int aperture_size=3 ); 采用 Canny 算法
做边缘检测image  输入图像 edges 输出的边缘图像 threshold1 第一个阈值threshold2  第二个阈值aperture_size   Sobel 算子内核大小
4、void cvCopy( const CvArr* src, CvArr* dst, const CvArr* mask=NULL ); 
在使用这个函数之前，你必须用cvCreateImage（）一类的函数先开一段内存，然后传递给dst。cvCopy会把src中的数据复制到dst的内存中。
5、cvCreateTrackbar  创建trackbar并将它添加到指定的窗口。
int cvCreateTrackbar( const char* trackbar_name, const char* window_name, int* value, int count, CvTrackbarCallback on_change ); 
trackbar_name  被创建的trackbar名字。window_name  窗口名字，这个窗口将为被创建trackbar的父对象。value  整数指针，它的值将反映滑块
的位置。这个变量指定创建时的滑块位置。count  滑块位置的最大值。最小值一直是0。on_change  每次滑块位置被改变的时候，被调用函数的指
针。这个函数应该被声明为void Foo(int);  如果没有回调函数，这个值可以设为NULL。函数cvCreateTrackbar用指定的名字和范围来创建trackbar
（滑块或者范围控制），指定与trackbar位置同步的变量，并且指定当trackbar位置被改变的时候调用的回调函数。被创建的trackbar显示在指定窗
口的顶端。*/


/*//-----------------------------------【头文件包含部分】---------------------------------------  
//            描述：包含程序所依赖的头文件  
//----------------------------------------------------------------------------------------------  
#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
  
//-----------------------------------【命名空间声明部分】---------------------------------------  
//            描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------  
using namespace cv;  
//-----------------------------------【main( )函数】--------------------------------------------  
//            描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main( )  
{  
    //载入原始图    
    Mat src = imread("emosue.jpg");  //工程目录下应该有一张名为1.jpg的素材图  
    Mat src1=src.clone();  
  
    //显示原始图   
    imshow("【原始图】Canny边缘检测", src);   
  
    //----------------------------------------------------------------------------------  
    //  一、最简单的canny用法，拿到原图后直接用。  
    //----------------------------------------------------------------------------------  
    Canny( src, src, 150, 100,3 );  
    imshow("【效果图】Canny边缘检测", src);   
  
      
    //----------------------------------------------------------------------------------  
    //  二、高阶的canny用法，转成灰度图，降噪，用canny，最后将得到的边缘作为掩码，拷贝原图到效果图上，得到彩色的边缘图  
    //----------------------------------------------------------------------------------  
    Mat dst,edge,gray;  
  
    // 【1】创建与src同类型和大小的矩阵(dst)  
    dst.create( src1.size(), src1.type() );  
  
    // 【2】将原图像转换为灰度图像  
    cvtColor( src1, gray, CV_BGR2GRAY );  
  
    // 【3】先用使用 3x3内核来降噪  
    blur( gray, edge, Size(3,3) );  
  
    // 【4】运行Canny算子  
    Canny( edge, edge, 3, 9,3 );  
  
    //【5】将g_dstImage内的所有元素设置为0   
    dst = Scalar::all(0);  
  
    //【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中  
    src1.copyTo( dst, edge);  
  
    //【7】显示效果图   
    imshow("【效果图】Canny边缘检测2", dst);   
  
  
    waitKey(0);   
  
    return 0;   
}  */
