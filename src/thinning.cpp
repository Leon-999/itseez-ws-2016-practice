#include "skeleton_filter.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

static void GuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & (m == 0))
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void GuoHallThinning(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC1 == src.type());

    dst = src / 255;

    cv::Mat prev = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat diff;

    do
    {
        GuoHallIteration(dst, 0);
        GuoHallIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

//
// Place optimized version here
//

int getCodeNeghbourhood(cv::Mat& source, int i, int j) {

    uchar p2 = source.at<uchar>(i-1, j);
    uchar p3 = source.at<uchar>(i-1, j+1);
    uchar p4 = source.at<uchar>(i, j+1);
    uchar p5 = source.at<uchar>(i+1, j+1);
    uchar p6 = source.at<uchar>(i+1, j);
    uchar p7 = source.at<uchar>(i+1, j-1);
    uchar p8 = source.at<uchar>(i, j-1);
    uchar p9 = source.at<uchar>(i-1, j-1);

    int code = (int) p2 * 1 +
               (int) p3 * 2 +
               (int) p4 * 4 +
               (int) p5 * 8 +
               (int) p6 * 16 +
               (int) p7 * 32 +
               (int) p8 * 64 +
               (int) p9 * 128;


    return code;
}

void generateMaskMatrixForGuoHall(uchar*  maskMatrix, int iter) {
    int p2, p3, p4, p5, p6, p7, p8, p9;  

    for(int code = 0; code < 256; ++code) 
    {
        p2 = ((code & (int)1) >> 0);
        p3 = ((code & (int)2) >> 1);
        p4 = ((code & (int)4) >> 2);
        p5 = ((code & (int)8) >> 3);
        p6 = ((code & (int)16) >> 4);
        p7 = ((code & (int)32) >> 5);
        p8 = ((code & (int)64) >> 6);
        p9 = ((code & (int)128) >> 7);

        int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                         (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
        int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
        int N  = N1 < N2 ? N1 : N2;
        int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

        if (C == 1 && (N >= 2 && N <= 3) & (m == 0)) 
            maskMatrix[code] = 1;
        else
            maskMatrix[code] = 0;     
    }
}

static void GuoHallIteration_optimized(cv::Mat& im, const uchar* maskMatrix)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            if(im.at<uchar>(i,j) != 0) {
                int code = getCodeNeghbourhood(im, i, j);

                marker.at<uchar>(i,j) = maskMatrix[code];
            }
        }
    }

    im &= ~marker;
}

void GuoHallThinning_optimized(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC1 == src.type());

    dst = src / 255;
    //dst = src.clone();

    cv::Mat prev = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat diff;

    uchar maskMatrix0[256];
    uchar maskMatrix1[256];
    generateMaskMatrixForGuoHall(maskMatrix0, 0);
    generateMaskMatrixForGuoHall(maskMatrix1, 1);

    do
    {
        GuoHallIteration_optimized(dst, maskMatrix0);
        GuoHallIteration_optimized(dst, maskMatrix1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

//
// Sample performance report
//
//           Name of Test               base          1           2          1          2
//                                                                           vs         vs
//                                                                          base       base
//                                                                       (x-factor) (x-factor)
// Thinning::Size_Only::640x480      333.442 ms  216.775 ms  142.484 ms     1.54       2.34
// Thinning::Size_Only::1280x720     822.569 ms  468.958 ms  359.877 ms     1.75       2.29
// Thinning::Size_Only::1920x1080    2438.715 ms 1402.072 ms 1126.428 ms    1.74       2.16
