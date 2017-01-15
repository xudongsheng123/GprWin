
#include "cv.h"
#include "highgui.h"

char wndname[] = "Edge";  
char tbarname[] = "Threshold"; 
int edge_thresh = 1;
IplImage *image = 0, *cedge = 0, *gray = 0, *edge = 0; 
// �����������callback ����
 
void on_trackbar(int h) 
{  
    cvSmooth( gray, edge, CV_BLUR, 3, 3, 0 ); 
    cvNot( gray, edge );  
    // �ԻҶ�ͼ����б�Ե���
    cvCanny(gray, edge, (float)edge_thresh, (float)edge_thresh*3, 3); 
    cvZero( cedge ); 
    // copy edge points  
    cvCopy( image, cedge, edge ); 
    // ��ʾͼ 
    cvShowImage(wndname, cedge); 
}  
int main( int argc, char** argv ) 
{  
    char* filename = argc == 2 ? argv[1] : (char*)"emosue.jpg";      
    if( (image = cvLoadImage( filename, 1)) == 0 ) 
        return -1;  
    // Create the output image
    cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);  
    // ����ɫͼ��ת��Ϊ�Ҷ�ͼ��
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
/*******�����к���˵��
1��cvSmooth���亯������Ϊ��
cvSmooth( const void* srcarr, void* dstarr, int smoothtype,int param1, int param2, double param3 )  
cvSmooth�����������Ƕ�ͼ�������ַ�����ͼ��ƽ�������У�srcarrΪ����ͼ��dstarrΪ���ͼ��param1Ϊƽ�������ĵ�һ��������
param2Ϊƽ�������ĵڶ������������param2ֵΪ0�����ʾ������Ϊparam1����param3�Ƕ�Ӧ��˹�����ı�׼�����smoothtype��ͼ��
ƽ���ķ���ѡ����Ҫ��ƽ���������������֣�CV_BLUR_NO_SCALE���򵥲����߶ȱ任��ģ��������ÿ��������param1��param2������͡�
CV_BLUR����ÿ��������param1��param2������Ͳ����߶ȱ任1/��param1?param2����CV_GAUSSIAN����ͼ����к˴�СΪparam1��param2
�ĸ�˹�����CV_MEDIAN����ͼ����к˴�СΪparam1��param1 ����ֵ�˲�����������Ƿ��ģ���CV_BILATERAL��˫���˲���Ӧ��˫�� 3x3 
�˲�����ɫ����Ϊparam1���ռ�����Ϊparam2��
2��void cvNot(const CvArr* src,CvArr* dst);  
����cvNot()�Ὣsrc�е�ÿһ��Ԫ�ص�ÿһλȡ����Ȼ��ѽ������dst����ˣ�һ��ֵΪ0x00��8λͼ�񽫱�ӳ�䵽0xff����ֵΪ0x83��ͼ��
����ӳ�䵽0x7c��
3��void cvCanny( const CvArr* image, CvArr* edges, double threshold1,double threshold2, int aperture_size=3 ); ���� Canny �㷨
����Ե���image  ����ͼ�� edges ����ı�Եͼ�� threshold1 ��һ����ֵthreshold2  �ڶ�����ֵaperture_size   Sobel �����ں˴�С
4��void cvCopy( const CvArr* src, CvArr* dst, const CvArr* mask=NULL ); 
��ʹ���������֮ǰ���������cvCreateImage����һ��ĺ����ȿ�һ���ڴ棬Ȼ�󴫵ݸ�dst��cvCopy���src�е����ݸ��Ƶ�dst���ڴ��С�
5��cvCreateTrackbar  ����trackbar��������ӵ�ָ���Ĵ��ڡ�
int cvCreateTrackbar( const char* trackbar_name, const char* window_name, int* value, int count, CvTrackbarCallback on_change ); 
trackbar_name  ��������trackbar���֡�window_name  �������֣�������ڽ�Ϊ������trackbar�ĸ�����value  ����ָ�룬����ֵ����ӳ����
��λ�á��������ָ������ʱ�Ļ���λ�á�count  ����λ�õ����ֵ����Сֵһֱ��0��on_change  ÿ�λ���λ�ñ��ı��ʱ�򣬱����ú�����ָ
�롣�������Ӧ�ñ�����Ϊvoid Foo(int);  ���û�лص����������ֵ������ΪNULL������cvCreateTrackbar��ָ�������ֺͷ�Χ������trackbar
��������߷�Χ���ƣ���ָ����trackbarλ��ͬ���ı���������ָ����trackbarλ�ñ��ı��ʱ����õĻص���������������trackbar��ʾ��ָ����
�ڵĶ��ˡ�*/


/*//-----------------------------------��ͷ�ļ��������֡�---------------------------------------  
//            ����������������������ͷ�ļ�  
//----------------------------------------------------------------------------------------------  
#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
  
//-----------------------------------�������ռ��������֡�---------------------------------------  
//            ����������������ʹ�õ������ռ�  
//-----------------------------------------------------------------------------------------------  
using namespace cv;  
//-----------------------------------��main( )������--------------------------------------------  
//            ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ  
//-----------------------------------------------------------------------------------------------  
int main( )  
{  
    //����ԭʼͼ    
    Mat src = imread("emosue.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ  
    Mat src1=src.clone();  
  
    //��ʾԭʼͼ   
    imshow("��ԭʼͼ��Canny��Ե���", src);   
  
    //----------------------------------------------------------------------------------  
    //  һ����򵥵�canny�÷����õ�ԭͼ��ֱ���á�  
    //----------------------------------------------------------------------------------  
    Canny( src, src, 150, 100,3 );  
    imshow("��Ч��ͼ��Canny��Ե���", src);   
  
      
    //----------------------------------------------------------------------------------  
    //  �����߽׵�canny�÷���ת�ɻҶ�ͼ�����룬��canny����󽫵õ��ı�Ե��Ϊ���룬����ԭͼ��Ч��ͼ�ϣ��õ���ɫ�ı�Եͼ  
    //----------------------------------------------------------------------------------  
    Mat dst,edge,gray;  
  
    // ��1��������srcͬ���ͺʹ�С�ľ���(dst)  
    dst.create( src1.size(), src1.type() );  
  
    // ��2����ԭͼ��ת��Ϊ�Ҷ�ͼ��  
    cvtColor( src1, gray, CV_BGR2GRAY );  
  
    // ��3������ʹ�� 3x3�ں�������  
    blur( gray, edge, Size(3,3) );  
  
    // ��4������Canny����  
    Canny( edge, edge, 3, 9,3 );  
  
    //��5����g_dstImage�ڵ�����Ԫ������Ϊ0   
    dst = Scalar::all(0);  
  
    //��6��ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��  
    src1.copyTo( dst, edge);  
  
    //��7����ʾЧ��ͼ   
    imshow("��Ч��ͼ��Canny��Ե���2", dst);   
  
  
    waitKey(0);   
  
    return 0;   
}  */
