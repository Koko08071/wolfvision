#include<iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include<time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

#define MAX_number 10

#define ROI_BIG 3000

#define MAX_VAL 3.0*pow(10,8)


struct HSV_SPACE_NUMBER
{
    int HSV_H_MIN;
    int HSV_S_MIN;
    int HSV_V_MIN;
    int HSV_H_MAX;
    int HSV_S_MAX;
    int HSV_V_MAX;
    int MAX;
};


//模版匹配(失败，每一张图的匹配度都差不多)，分割ROI加拐点化直线判断交点

//摄像头
VideoCapture capture(2);
HSV_SPACE_NUMBER parameter1;
Mat src,dst,ds,src1,dst1,Dsrc;
Mat ima,sr,sr1;
Mat I;
Mat ROI,ROI_2,ROI_3;
int value_min,value_max;
Rect num;
Rect zxc;
int structElementSize[3] = {1,0,10};
Mat element[3];
Mat ROI_[4];
clock_t start,finish;
int b_val;
int asd;
int thre;
int thre_max;
Mat srcImage;
vector<Vec3f>circles;
Mat templateImage;
Mat resultImage;
double area=0.0;


void initial();
void Image_preprocessing();
void Task_two();
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst);
void processFrame(Mat & img, Rect & rect);


void processFrame(Mat & img, Rect & rect)//绘制外接矩形
{
 //寻找外接轮廓
 vector<vector<Point>>contours;
 vector<Vec4i>hierarchy;
 findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
   
 if (contours.size() > 0)
 {
     for (size_t i = 0; i < contours.size(); i++)
     {
         double contours_Area = contourArea(contours[static_cast<int>(i)]);//面积
         rect = boundingRect(contours[static_cast<int>(i)]);//外接矩形
         if (contours_Area > area)
         {
             area = contours_Area;
         }
     }
       
 }
 else
 {
     rect.x = rect.y = rect.width = rect.height = 0;
 }
}

// 1.细化骨架
void cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
{
    //http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
    //算法有问题，得不到想要的效果

    if(src.type()!=CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }


    //非原地操作时候，copy src到dst
    if(dst.data!=src.data)
    {
        src.copyTo(dst);
    }
 
    

    ////细化提取骨架
    int i, j;
    int width, height;

    //之所以减2，是方便处理8邻域，防止越界
    width = src.cols -2;
    height = src.rows -2;
    int step = src.step;
    int  p2,p3,p4,p5,p6,p7,p8,p9;
    uchar* img;
    bool ifEnd;
    int A1;
    cv::Mat tmpimg;


    while(1)
    {
        dst.copyTo(tmpimg);
        ifEnd = false;
        img = tmpimg.data+step;
        for(i = 2; i < height; i++)
        {
            img += step;
            for(j =2; j<width; j++)
            {
                uchar* p = img + j;
                A1 = 0;
                if( p[0] > 0)
                {
                    if(p[-step]==0&&p[-step+1]>0) //p2,p3 01模式
                    {
                        A1++;
                    }

                    if(p[-step+1]==0&&p[1]>0) //p3,p4 01模式
                    {
                        A1++;
                    }

                    if(p[1]==0&&p[step+1]>0) //p4,p5 01模式
                    {
                        A1++;
                    }

                    if(p[step+1]==0&&p[step]>0) //p5,p6 01模式
                    {
                        A1++;
                    }

                    if(p[step]==0&&p[step-1]>0) //p6,p7 01模式
                    {
                        A1++;
                    }

                    if(p[step-1]==0&&p[-1]>0) //p7,p8 01模式
                    {
                        A1++;
                    }

                    if(p[-1]==0&&p[-step-1]>0) //p8,p9 01模式
                    {
                        A1++;
                    }

                    if(p[-step-1]==0&&p[-step]>0) //p9,p2 01模式
                    {
                        A1++;
                    }

                    p2 = p[-step]>0?1:0;
                    p3 = p[-step+1]>0?1:0;
                    p4 = p[1]>0?1:0;
                    p5 = p[step+1]>0?1:0;
                    p6 = p[step]>0?1:0;
                    p7 = p[step-1]>0?1:0;
                    p8 = p[-1]>0?1:0;
                    p9 = p[-step-1]>0?1:0;

                    //计算AP2,AP4
                    int A2, A4;
                    A2 = 0;
                    //if(p[-step]>0)
                    {
                        if(p[-2*step]==0&&p[-2*step+1]>0) A2++;
                        if(p[-2*step+1]==0&&p[-step+1]>0) A2++;
                        if(p[-step+1]==0&&p[1]>0) A2++;
                        if(p[1]==0&&p[0]>0) A2++;
                        if(p[0]==0&&p[-1]>0) A2++;
                        if(p[-1]==0&&p[-step-1]>0) A2++;
                        if(p[-step-1]==0&&p[-2*step-1]>0) A2++;
                        if(p[-2*step-1]==0&&p[-2*step]>0) A2++;
                    }
 
 
                    A4 = 0;
                    //if(p[1]>0)
                    {
                        if(p[-step+1]==0&&p[-step+2]>0) A4++;
                        if(p[-step+2]==0&&p[2]>0) A4++;
                        if(p[2]==0&&p[step+2]>0) A4++;
                        if(p[step+2]==0&&p[step+1]>0) A4++;
                        if(p[step+1]==0&&p[step]>0) A4++;
                        if(p[step]==0&&p[0]>0) A4++;
                        if(p[0]==0&&p[-step]>0) A4++;
                        if(p[-step]==0&&p[-step+1]>0) A4++;
                    }
 
 
                    //printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
                    //printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
                    if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7  &&  A1==1)
                    {
                        if(((p2==0||p4==0||p8==0)||A2!=1)&&((p2==0||p4==0||p6==0)||A4!=1)) 
                        {
                            dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
                            ifEnd = true;
                            //printf("\n");
 
                            //PrintMat(dst);
                        }
                    }
                }
            }
        }
        //printf("\n");
        //PrintMat(dst);
        //PrintMat(dst);
        //已经没有可以细化的像素了，则退出迭代
        if(!ifEnd) break;
    }
}


//2。匹配绿色
double macth_green()
{
    //模版匹配：匹配图像、映射图像、映射图像长、宽
    Mat greed_right1;
    Mat g_resultImage;
    int resultImage_cols, resultImage_rows;
    //映射图像初始化
        resultImage_cols = ROI.cols - ROI.cols +1;
        resultImage_rows = ROI.rows - ROI.rows +1;
        g_resultImage.create(resultImage_cols, resultImage_rows, CV_32FC1);

        //修改匹配图像大小，与感兴趣区域大小相同
        resize(I,greed_right1,ROI.size());

        //模版匹配
        matchTemplate(ROI,greed_right1,g_resultImage,TM_CCORR);
        //映射图像 解析数据
        double minVal; 
	    double maxVal; 
	    Point minLoc; 
	    Point maxLoc; 
        
	
	    minMaxLoc( g_resultImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat() ); /// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好 
        return maxVal;
}


//摄像头帧数
void Frame_rate()
{
    // double wkj_time = double(finish-start)/CLOCKS_PER_SEC*1000;
    double wkj_Frame = 1000/(double(finish-start)/CLOCKS_PER_SEC*1000);
    // cout<<"算法进行时间为："<<time<<endl;
    // cout<<"视觉识别帧率为："<<Frame<<endl;
    std::string str = std::to_string(wkj_Frame);
    string Frame_name = "FPS:";
    Frame_name +=str;
    //cout<<"视觉识别帧率为："<<Frame_name<<endl;
    putText(Dsrc,Frame_name,Point(0,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
}

void on_track(int ,void*)
{

    
}


void initial()
{
    I = imread("/home/zjk/Pictures/IMG_20220118_133750.jpg");
    b_val=1;
    asd =1;
    thre = 77;
    thre_max = 155;


}

void Image_preprocessing()
{
    //分离通道参数
    vector<Mat>image;


    blur(Dsrc,src,Size(2*b_val+1,2*b_val+1));
    split(src,image);

    //蓝红通道相减得出的二值化图像         顺序为：BGR
    dst = image[0] - image[2];     //蓝减红
    //imshow("dsm",dst);
    threshold(dst,dst,thre,thre_max,THRESH_BINARY);
   // imshow("dst",dst); 

  
    //内核
    element[0] = getStructuringElement(MORPH_RECT,Size(2*structElementSize[0]+1,2*structElementSize[0]+1));
    element[1] = getStructuringElement(MORPH_RECT,Size(2*structElementSize[1]+1,2*structElementSize[1]+1));
    element[2] = getStructuringElement(MORPH_RECT,Size(2*structElementSize[2]+1,2*structElementSize[2]+1));
    
    //形态学操作
    morphologyEx(dst,ima, MORPH_OPEN, element[0]);
    morphologyEx(ima,sr, MORPH_DILATE, element[1]);
    morphologyEx(sr,ds, MORPH_CLOSE, element[2]);

  //  imshow("ds",ds);
    


}


//任务二关键识别逻辑代码
void Task_two()
{
    //需要变量
    //寻找轮廓参数
    vector<vector<Point>>v1;
    vector<Vec4i>hierarchy;
    //绘制矩形
    Rect num;
    //感兴趣区域
    Mat roi;
    //模版匹配数值
    double maxVal; 
    
    //vector<Vec4i>lines;
    
    //选择算法变量：编号、最大值
    int numbers = -1;
    int Max = 0;

    //寻找轮廓
    findContours(ds,v1,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);
    if(hierarchy.size()==0)
    {
        return ;
    }

    //选择最大轮廓面积
    for(size_t i =0;hierarchy.size()>i;i++)
    {
        num = boundingRect(v1[i]);
        if(Max<num.area())
        {
            numbers = i;
            Max = num.area();
        }
    }

    cout<<"MAX： "<<Max<<endl;
    //淘汰面积不大的轮廓

    if(Max<ROI_BIG)
    {
        return;
    }

    num = boundingRect(v1[numbers]);
    roi = Dsrc(num);
    ROI = roi.clone();    //克隆
    maxVal = macth_green();
    cout<<maxVal<<endl;

  //  imshow("JKL",ROI);               //框箭头

    if((maxVal<MAX_VAL))
    {
        return;
    }
    //先排除反光现象
    //imshow("ROI",ROI);
    

    vector<Mat>image;
    split(ROI,image);
    ROI_2 = image[0] - image[2];
    blur(ROI_2,ROI_2,Size(b_val+1,b_val+1));
    threshold(ROI_2,ROI_2,thre,thre_max,THRESH_BINARY_INV);

  //  imshow("UOI",ROI_2);       //二值化箭头
    
    v1.clear();
    hierarchy.clear();

    //寻找轮廓
    findContours(ROI_2,v1,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);
    if(hierarchy.size()==0)
    {
        return;
    }

    numbers = -1;
    Max = 0;

    //选择最大轮廓面积
    for(size_t i =0;hierarchy.size()>i;i++)
    {
        num = boundingRect(v1[i]);
        if(Max<num.area())
        {
            if(1.0*num.height/num.width>1.5||1.0*num.width/num.height>1.5)
            {
                numbers = i;
                Max = num.area();
            }
            
        }
    }


    cout<<"Maxdddd:"<<Max<<endl;

    if(Max<500)
    {
        return;
    }

    if(numbers!=-1)
    {
        int number[2]={0};   //白点数量
        
        num = boundingRect(v1[numbers]);
        if(1.0*num.height/num.width>1.5)
        {
            rectangle(roi,Rect(0,0,roi.cols,roi.rows),Scalar(0,255,0),3,8,0);
            cout<<"直行"<<endl;   
            return;
        }

        ROI_3 = ROI_2(num);
        ROI_[0] = ROI_3.clone();   //原图
       
        cvHilditchThin1(ROI_[0],ROI_[1]);    //细化提取骨架
        
        //将箭头一分为二
        ROI_[2] = ROI_[1](Rect(0,0,ROI_[1].cols/2-4,ROI_[1].rows));       //左
        ROI_[3] = ROI_[1](Rect(ROI_[1].cols/2-2,0,ROI_[1].cols/2-4,ROI_[1].rows));   //右
       

        //遍历ROI区域，计算符合条件的像素点区域 
        for(int o=2;o<4;o++)
        {
            for(int i=0;i<ROI_[2].rows;i++)
            {
                uchar* data = ROI_[o].ptr<uchar>(i);
                for(int p=0;p<ROI_[2].cols;p++)
                {
                
                    if(data[p]>0)
                    {
                        //number[0]/[1]
                        number[o-2]++;
                    }
            
                }
            }
        }


        cout<<number[0]<<":"<<number[1]<<endl;
                
        //根据白色像素点来判断左右
        //number[0]：为左半图白色像素点
        //number[1]：为右半图白色像素点
        if(number[0]>number[1])     //左>右
        {
            rectangle(roi,Rect(0,0,roi.cols,roi.rows),Scalar(0,255,0),3,8,0);
            cout<<"左转"<<endl;
        }

        if(number[0]<number[1])     //右>左
        {
            rectangle(roi,Rect(0,0,roi.cols,roi.rows),Scalar(0,255,0),3,8,0);
            cout<<"右转"<<endl;   
        }
    }
   
}



int main()
{   
    initial();

    namedWindow("parameter",WINDOW_NORMAL);
    createTrackbar("MORPH_OPEN","parameter",&structElementSize[0],100,on_track);
    createTrackbar("MORPH_DILATE","parameter",&structElementSize[1],100,on_track);
    createTrackbar("MORPH_CLOSE","parameter",&structElementSize[2],100,on_track);
    createTrackbar("b_val","parameter",&b_val,100,on_track);
    createTrackbar("asd","parameter",&asd,100,on_track);
    createTrackbar("THRE","parameter",&thre,254,on_track);
    createTrackbar("THREd","parameter",&thre_max,255,on_track);

     Rect roi;//存储最大外接矩形的数据
   
//  VideoCapture capture;
//  capture.open(0);

 if (!capture.isOpened())
 {
     cout << "图像读取错误！～" << endl;
     return -1;
 }
   
 Mat frame,dst;
 Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
 Mat kernel_dilite = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));

 while (capture.read(frame))
 {
        start = clock();

        //分离通道参数
        vector<Mat>image;
        split(frame,image);
        //绿蓝通道相减得出绿灯的二值化图像
        dst = image[1] - image[0];          
     //开操作去噪点
        morphologyEx(dst, dst, MORPH_OPEN, kernel, Point(-1, -1), 1);
        threshold(dst,dst,50,255,THRESH_BINARY);
     //膨胀操作把绿灯具体化的显示出来
        dilate(dst, dst, kernel_dilite, Point(-1, -1), 2);
     //   imshow("output video", dst);
        processFrame(dst, roi);
        rectangle(frame, roi, Scalar(0, 0, 255), 3, 8, 0);
        
        
        if(area>=2000)  //根据距离调参数
        {
            int a;
            cout<<"绿灯识别完成"<<endl;
            cout<<"area:"<<area<<endl;
            a++;
            if(a>=10) break;
        }
        finish= clock();

        double abc_Frame = 1000/(double(finish-start)/CLOCKS_PER_SEC*1000);
        std::string str = std::to_string(abc_Frame);
        string Frame_name = "FPS:";
        Frame_name +=str;
        cout<<"视觉识别帧率为："<<Frame_name<<endl;
        putText(frame,Frame_name,Point(0,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
        imshow("input video", frame);

        // capture>>Dsrc;
        // Image_preprocessing();
        // Task_two();
        // Frame_rate();
        // imshow("SRC",Dsrc);     // 摄像头画面

     char c = waitKey(50);
     if (c == 27)
     {
         break;
     }
 }

    while(1)
    {
        capture>>Dsrc;
        start=clock();
        Image_preprocessing();
        Task_two();
        finish=clock();
        Frame_rate();
        imshow("SRC",Dsrc);     // 摄像头画面
        waitKey(1);
    }

    capture.release();
    return 0;
}
