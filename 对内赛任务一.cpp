// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/highgui/highgui.hpp>
// #include <opencv4/opencv2/imgproc/imgproc.hpp>
// #include<iostream>
// #include <stdlib.h>    　//定义杂项函数及内存分配函数
// #include<time.h>

// using namespace std;
// using namespace cv;


// //hsv参数
// struct HSV_SPACE_NUMBER
// {
//     int HSV_H_MIN;
//     int HSV_S_MIN;
//     int HSV_V_MIN;
//     int HSV_H_MAX;
//     int HSV_S_MAX;
//     int HSV_V_MAX;
//     int MAX;
// };

// VideoCapture capture(0);
// Mat image;
// Mat ROI;
// Mat image_key[2];                 //关键帧  （不可用）
// HSV_SPACE_NUMBER parameter1;
// bool one_or_two;                    //判断
// vector<Vec3f>circles;
// int Number_of_successful=0;         //帧数参数
// clock_t start,finish;               //程序运行时间



// void on_track(int,void*)            //调试范围  （调制外接摄像头可用）
// {

// }

// void Task_one()
// {
//     //定义所需变量

//     Mat ROI_1;
//     int Point_count=0;              //白点的数量
//     int number=-1;                  //圆的位置
//     int max=-1;                     //白点数量的最大值
//     //开始识别

//     for(size_t i = 0;i<circles.size();i++)
//     {

//         //防止边界问题
//         auto it = *(circles.begin()+i);
//         //     //auto可以在声明变量的时候根据变量初始值的类型自动为此变量选择匹配的类型

//         double x1 = it[0]-it[2];
//         double x2 = it[0]+it[2];
//         double y1 = it[1]-it[2];
//         double y2 = it[1]+it[2];

//         if(x1 < 0)
//         {
//             x1 = 0;
//         }
//         if(x2>image.cols)
//         {
//             x2 = image.cols;
//         }
//         if(y1<0)
//         {
//             y1 = 0;
//         }
//         if(y2 >image.rows)
//         {
//             y2 = image.rows;
//         }


//         //定义感兴趣区间，HSV颜色转换和二值化
//         ROI = image(Rect(Point(x1,y1),Point(x2,y2)));
//         cv::cvtColor(ROI,ROI_1,COLOR_BGR2HSV);

//         //inRange二值化
//         cv::inRange(ROI_1,
//         Scalar(parameter1.HSV_H_MIN,parameter1.HSV_S_MIN,parameter1.HSV_V_MIN),
//         Scalar(parameter1.HSV_H_MAX,parameter1.HSV_S_MAX,parameter1.HSV_V_MAX),
//         ROI_1);

        
//         //随机取点判断是否为绿灯图像
//         Point_count=0;
//         for(int i = 0;i<100;i++)
//         {
//             int x = rand()%ROI_1.cols-1;
//             int y = rand()%ROI_1.rows-1;
//             if(ROI_1.at<uchar>(y,x)!= 0)
//             {
//                 Point_count++;
//             }
//         }
        
//         //判断成功后
//         if(Point_count>50)
//         {
//             if(Point_count>max)
//             {
//                 max = Point_count;
//                 number = i;
//             }
//         }
//     }


//     //出现绿灯后执行
//     if(number>=0)
//     {
//         //准备合适的参数
//         auto it = circles[number];
//         double x1 = it[0]-it[2];
//         double x2 = it[0]+it[2];
//         double y1 = it[1]-it[2];
//         double y2 = it[1]+it[2];
        
//         //对目标区域画框
//         Rect green_right = Rect(Point(x1,y1),Point(x2,y2));
//         rectangle(image,green_right,Scalar(0,255,0),9);
//         putText(image,"green",Point(x1,y1),FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,0));     //在图片上写入字
//         Number_of_successful++;

//         //判断任务一完成
//         if(Number_of_successful>=5)
//         {
//             one_or_two = true;
//             image_key[0]= image.clone();
//             cout<<"绿灯完成"<<endl;
//             cout<<"开始启动"<<endl;
//         }
        

//     }
// }

// void Task_two()
// {

// }


// int main()
// {
    
//     parameter1.HSV_H_MAX=255;
//     parameter1.HSV_H_MIN=0;
//     parameter1.HSV_S_MAX=150;
//     parameter1.HSV_S_MIN=70;
//     parameter1.HSV_V_MAX=255;
//     parameter1.HSV_V_MIN=0;
//     parameter1.MAX=255;
//     capture>>image;
//     one_or_two = false;

//      //窗口设置
//     namedWindow("Camer",WINDOW_NORMAL);
//     namedWindow("parameter",WINDOW_NORMAL);
    

//     createTrackbar("H_MIN","parameter",&parameter1.HSV_H_MIN,parameter1.MAX,on_track);
//     createTrackbar("S_MIN","parameter",&parameter1.HSV_S_MIN,parameter1.MAX,on_track);
//     createTrackbar("V_MIN","parameter",&parameter1.HSV_V_MIN,parameter1.MAX,on_track);
//     createTrackbar("H_MAX","parameter",&parameter1.HSV_H_MAX,parameter1.MAX,on_track);
//     createTrackbar("S_MAX","parameter",&parameter1.HSV_S_MAX,parameter1.MAX,on_track);
//     createTrackbar("V_MAX","parameter",&parameter1.HSV_V_MAX,parameter1.MAX,on_track);
//     //循环
//     while(1)
//     {
//         start=clock();                    //程序运行开始时间
//         capture>>image;                   //开启摄像头读取图片

//         Mat src;
//         cv::cvtColor(image,src,COLOR_BGR2GRAY);

//         HoughCircles(src,circles,HOUGH_GRADIENT,1.5,100,200,100,0,0);


//         if(one_or_two)                     //若为false则 跳转到任务二
//         {
//             Task_two();
//             imshow("parameter",image_key[0]);
//         }
        
//         else
//         {
//             Task_one();
//         }
 
//         finish=clock();                   //程序运行结束时间

//         double wkj_Frame = 1000/(double(finish-start)/CLOCKS_PER_SEC*1000);
//         // cout<<"算法进行时间为："<<time<<endl;
//         // cout<<"视觉识别帧率为："<<Frame<<endl;
//         std::string str = std::to_string(wkj_Frame);
//         string Frame_name = "FPS:";
//         Frame_name +=str;
//         //cout<<"视觉识别帧率为："<<Frame_name<<endl;
//         putText(image,Frame_name,Point(0,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
        

//         imshow("Camer",image);
        
//         waitKey(1);
//     }
    
//     waitKey(0);
//     return 0;
// }










// // ////第二版任务一（已完成）
// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/highgui/highgui.hpp>
// #include <opencv4/opencv2/imgproc/imgproc.hpp>
// #include<iostream>
// #include <time.h>

// using namespace cv;
// using namespace std;

// clock_t start,finish;
// Mat srcImage;
// vector<Vec3f>circles;
// Mat templateImage;
// Mat resultImage;
// double area=0.0;

// void processFrame(Mat & img, Rect & rect)//绘制外接矩形
// {
//  //寻找外接轮廓
//  vector<vector<Point>>contours;
//  vector<Vec4i>hierarchy;
//  findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
   
//  if (contours.size() > 0)
//  {
//      for (size_t i = 0; i < contours.size(); i++)
//      {
//          double contours_Area = contourArea(contours[static_cast<int>(i)]);//面积
//          rect = boundingRect(contours[static_cast<int>(i)]);//外接矩形
//          if (contours_Area > area)
//          {
//              area = contours_Area;
//          }
//      }
       
//  }
//  else
//  {
//      rect.x = rect.y = rect.width = rect.height = 0;
//  }
// }


// int main(int argc, char** argv)
// {
//  Rect roi;//存储最大外接矩形的数据
   
//  VideoCapture capture;
//  capture.open(0);

//  if (!capture.isOpened())
//  {
//      cout << "图像读取错误！～" << endl;
//      return -1;
//  }
   
//  Mat frame,dst;
//  Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
//  Mat kernel_dilite = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));

//  while (capture.read(frame))
//  {
//      start = clock();
//      //筛选出绿色
//      //inRange(frame, Scalar(0, 127, 0), Scalar(120, 255, 120), dst);

//         //分离通道参数
//         vector<Mat>image;
//         split(frame,image);
//         //绿蓝通道相减得出绿灯的二值化图像
//         dst = image[1] - image[0];          
//      //开操作去噪点
//      morphologyEx(dst, dst, MORPH_OPEN, kernel, Point(-1, -1), 1);
//         threshold(dst,dst,50,255,THRESH_BINARY);
//      //膨胀操作把绿灯具体化的显示出来
//      dilate(dst, dst, kernel_dilite, Point(-1, -1), 2);
//      imshow("output video", dst);
//      processFrame(dst, roi);
//      rectangle(frame, roi, Scalar(0, 0, 255), 3, 8, 0);
//         if(area>=800)  //根据距离调参数
//         {
//             cout<<"绿灯识别完成"<<endl;
//             cout<<"area:"<<area<<endl;
//         }
//         finish= clock();

//         double abc_Frame = 1000/(double(finish-start)/CLOCKS_PER_SEC*1000);
//         std::string str = std::to_string(abc_Frame);
//         string Frame_name = "FPS:";
//         Frame_name +=str;
//         cout<<"视觉识别帧率为："<<Frame_name<<endl;
//         putText(frame,Frame_name,Point(0,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
//      imshow("input video", frame);

//      char c = waitKey(50);
//      if (c == 27)
//      {
//          break;
//      }
//  }

//  capture.release();
//  waitKey(0);
//  return 0;

// }











////kj版
// #include<iostream>
// #include<opencv2/opencv.hpp>
// #include<time.h>
// #include <stdlib.h>
// #include<cmath>

// using namespace std;
// using namespace cv;

// #define ROI_BIG 3000

// #define POINT_NUMBER 50

// #define MAX_VAL 2.0*pow(10,8)

// //摄像头
// VideoCapture capture(0);

// Mat src,dst,ds;
// Mat greed_right;
// Mat ROI;
// Mat ROI_1;
// Mat ROI_2;

// Rect zxc;
// int value_min,value_max;
// clock_t start,finish;


// void initial();
// void Image_preprocessing();
// void Task_one();
// double getDistance(Point A,Point B);
// void on_track(int ,void*);

// double macth_green();
// int point_macth();
// bool analysis(Rect&num,double&maxVal,int&Point_count);



// double macth_green()
// {
//     //模版匹配：匹配图像、映射图像、映射图像长、宽
//     Mat greed_right1;
//     Mat g_resultImage;
//     int resultImage_cols, resultImage_rows;
//     //映射图像初始化
//         resultImage_cols = ROI.cols - ROI.cols +1;
//         resultImage_rows = ROI.rows - ROI.rows +1;
//         g_resultImage.create(resultImage_cols, resultImage_rows, CV_32FC1);

//         //修改匹配图像大小，与感兴趣区域大小相同
//         resize(greed_right,greed_right1,ROI.size());

//         //模版匹配
//         matchTemplate(ROI,greed_right1,g_resultImage,TM_CCORR);
//         //映射图像 解析数据
//         double minVal; 
// 	    double maxVal; 
// 	    Point minLoc; 
// 	    Point maxLoc; 
        
	
// 	    minMaxLoc( g_resultImage, &minVal, &maxVal, &minLoc, &maxLoc, Mat() ); /// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好 
//         return maxVal;
// }
// int point_macth()
// {
//     int Point_count =0;
//     for(int i = 0;i<POINT_NUMBER;i++)
//     {
//         int x = rand()%ROI_2.cols-1;
//         int y = rand()%ROI_2.rows-1;
//         if(ROI_1.at<uchar>(y,x)!= 0)
//         {
//             Point_count++;
//         }
//     }
//     return Point_count;
// }
// bool analysis(Rect&num,double&maxVal,int&Point_count)
// {
//     //两点间距离
//     double dis;
//     if(zxc.width!=0)
//     {
//         dis = getDistance(Point(num.x,num.y),Point(zxc.x,zxc.y));
//     }
//     if(maxVal>MAX_VAL&&Point_count>=POINT_NUMBER/2)
//     {
//         //二重判断舍去矩形位移差太大的图像
//         if(dis>200)
//         {
//             zxc = Rect(0,0,0,0);
//             return false;
//         }
//         else{
//             return true;
//         }
        
//     }
    
// }

// //获取点间距离
// double getDistance(Point A,Point B)
// {
//     double dis;
//     dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
//     return sqrt(dis);
// }

// void on_track(int ,void*)
// {
   
//     //imshow("parameter",ds);
// }
// void initial()
// {
//     capture>>src;
//     value_min = 30;
//     value_max = 100;
//     greed_right = imread("/home/zjk/Pictures/IMG_20220118_133750.jpg");

// }
// void Image_preprocessing()
// {
//     //分离通道参数
//     vector<Mat>image;
//     split(src,image);
//     //绿蓝通道相减得出绿灯的二值化图像
//     dst = image[1] - image[0];
//     //imshow("4",dst);

//     //加深图像色差
//     threshold(dst,dst,50,255,THRESH_BINARY);
//     //imshow("5",dst);
//     //waitKey(0);
    
//     //避免灯光导致屏幕反光而造成误差
//     Mat Element;
//     Element = getStructuringElement(MORPH_RECT,Size(value_min+1,value_min+1));
//     dilate(dst,ds,Element);
//     //waitKey(0);
// }
// void Task_one()
// {
//     //需要变量
//     //寻找轮廓参数
//     vector<vector<Point>>v1;
//     vector<Vec4i>hierarchy;
//     //绘制矩形
//     Rect num;
//     //感兴趣区域
//     Mat roi;
//     //判断
//     bool x = false;
//     //判断像素点、符合点参数
//     int Point_count;
    
//     //模版匹配数值
//     double maxVal; 
//     //原点（0，0）
// 	Point matchLoc = Point(0,0); 

//     //选择算法变量：编号、最大值
//     int numbers = -1;
//     int Max = 0;
    
//     //寻找轮廓
//     findContours(ds,v1,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);
//     if(hierarchy.size()==0)
//     {
//         return ;
//     }
//     //选择最大轮廓面积
//     for(size_t i =0;hierarchy.size()>i;i++)
//     {
//         num = boundingRect(v1[i]);
//         if(Max<num.area())
//         {
//             numbers = i;
//             Max = num.area();
//         }
//     }
//     //淘汰面积不大的轮廓
//     if(Max<ROI_BIG)
//     {
//         numbers = -1;
//     }
//     if(numbers!=-1)
//     {
//         num = boundingRect(v1[numbers]);
//         zxc = num;
//         //感兴趣区域赋值
//         roi = src(num);
//         ROI_1 = ds(num);
//         ROI_2 = ROI_1.clone();
//         ROI = roi.clone();
//         maxVal = macth_green();
//         //cout<<maxVal<<endl;
       
//         //像素点匹配
//         Point_count = point_macth();
//         //cout<<Point_count<<endl;
//         x = analysis(num,maxVal,Point_count);
//         if(x)
//         {
//             //cout<<"x"<<endl;
//             // 通过判断后将ROI区域框出
//             rectangle( roi, matchLoc, Point( matchLoc.x + ROI.cols , matchLoc.y + ROI.rows ), Scalar(0,0,255), 2, 8); 
//             putText(src ,"green",Point(num.x,num.y),FONT_HERSHEY_COMPLEX,0.5,Scalar(255,0,0));
//             cout<<"绿灯"<<endl;
//         }
//     }

    
// }
// void Frame_rate()
// {
//     // double wkj_time = double(finish-start)/CLOCKS_PER_SEC*1000;
//     double wkj_Frame = 1000/(double(finish-start)/CLOCKS_PER_SEC*1000);
//     // cout<<"算法进行时间为："<<time<<endl;
//     // cout<<"视觉识别帧率为："<<Frame<<endl;
//     std::string str = std::to_string(wkj_Frame);
//     string Frame_name = "FPS:";
//     Frame_name +=str;
//     //cout<<"视觉识别帧率为："<<Frame_name<<endl;
//     putText(src,Frame_name,Point(0,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
// }


// int main()
// {
//     initial();
//     namedWindow("parameter",WINDOW_NORMAL);
//     namedWindow("kk",WINDOW_NORMAL);
//     createTrackbar("value","parameter",&value_min,value_max,on_track);
//     while(1)
//     {
//         capture>>src;
//         start=clock();

//         Image_preprocessing();
//         Task_one();
//         finish=clock();
//         Frame_rate();

//         imshow("kk",src);
//         waitKey(1);
//     }
    
//     return 0;
// }