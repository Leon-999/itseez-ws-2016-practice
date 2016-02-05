#include "skeleton_filter.hpp"

#if defined __SSSE3__  || (defined _MSC_VER && _MSC_VER >= 1500)
#  include "tmmintrin.h"
#  define HAVE_SSE
#endif

#include <string>
#include <sstream>
#include <cmath>

// Function for debug prints
template <typename T>
std::string __m128i_toString(const __m128i var)
{
    std::stringstream sstr;
    const T* values = (const T*) &var;
    if (sizeof(T) == 1)
    {
        for (unsigned int i = 0; i < sizeof(__m128i); i++)
        {
            sstr << (int) values[i] << " ";
        }
    }
    else
    {
        for (unsigned int i = 0; i < sizeof(__m128i) / sizeof(T); i++)
        {
            sstr << values[i] << " ";
        }
    }
    return sstr.str();
}

void ConvertColor_BGR2GRAY_BT709(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC3 == src.type());
    cv::Size sz = src.size();
    dst.create(sz, CV_8UC1);

    const int bidx = 0;

    for (int y = 0; y < sz.height; y++)
    {
        const cv::Vec3b *psrc = src.ptr<cv::Vec3b>(y);
        uchar *pdst = dst.ptr<uchar>(y);

        for (int x = 0; x < sz.width; x++)
        {
            float color = 0.2126 * psrc[x][2-bidx] + 0.7152 * psrc[x][1] + 0.0722 * psrc[x][bidx];
            pdst[x] = (int)(color + 0.5);
        }
    }
}

#define _2_IN_16 (25536)
#define _10_IN_4 (1000)

unsigned int getFixPoint(float floatValue) {
    unsigned int fixValue = (int)(floatValue * _10_IN_4);

    return fixValue;
}

unsigned int getFixPoint(uchar ucharValue)
{
    unsigned int fixValue = (int)ucharValue * _10_IN_4;

    return fixValue;
}

unsigned int getFixPoint(int ucharValue)
{
    unsigned int fixValue = ucharValue * _10_IN_4;

    return fixValue;
}

float getFloatPoint(unsigned int fixValue)
{
    unsigned int beforePoint = fixValue / _10_IN_4;
    unsigned int afterPoint = fixValue - beforePoint * _10_IN_4;

    float m = 0.001f;

    float floatValue = beforePoint + (float)afterPoint * m;

    return floatValue;
}

unsigned int sumFixPoint(unsigned int first, unsigned int second)
{
    return first + second;
}

unsigned int multiplyFixPoint(unsigned int first, unsigned int second)
{
    int multiplication = first * second / _10_IN_4;

    return multiplication;
}


void ConvertColor_BGR2GRAY_BT709_fpt(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC3 == src.type());
    cv::Size sz = src.size();
    dst.create(sz, CV_8UC1);

    unsigned int redC = getFixPoint(0.2126f);
    unsigned int greenC = getFixPoint(0.7152f);
    unsigned int blueC = getFixPoint(0.0722f);

    float rC = getFloatPoint(redC);
    redC = sumFixPoint(redC, blueC);
    redC = multiplyFixPoint(redC, getFixPoint(2));

    float test = 0.4f;
    int fixTest = getFixPoint(test);
    float floatTest = getFloatPoint(fixTest);

    const int bidx = 0;

    for (int y = 0; y < sz.height; y++)
    {
        const cv::Vec3b *psrc = src.ptr<cv::Vec3b>(y);
        uchar *pdst = dst.ptr<uchar>(y);

        for (int x = 0; x < sz.width; x++)
        {
            unsigned int red = getFixPoint(psrc[x][2-bidx]);
            unsigned int green = getFixPoint(psrc[x][1]);
            unsigned int blue = getFixPoint(psrc[x][bidx]);

            unsigned int sR = multiplyFixPoint(redC, red);
            unsigned int sG = multiplyFixPoint(greenC, green);
            unsigned int sB = multiplyFixPoint(blueC, blue);
            unsigned int fixColor = sumFixPoint(sR, sG);
            fixColor = sumFixPoint(fixColor, sB);
            
            float color = 0.2126 * psrc[x][2-bidx] + 0.7152 * psrc[x][1] + 0.0722 * psrc[x][bidx];
            //color = modf(
            float myColor = getFloatPoint(fixColor);

            int i;
            if(color - myColor != 0)
                myColor = getFloatPoint(fixColor);

            ///myColor =my
            uchar col =  (int)(color + 0.5);
            uchar myCol = (int)(myColor + 0.5);

            if(col != myCol)
                 myColor = getFloatPoint(fixColor);

            //pdst[x] = (int)(color + 0.5);
            pdst[x] = (int)(myColor + 0.5);
        }
    }
}

void ConvertColor_BGR2GRAY_BT709_simd(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC3 == src.type());
    cv::Size sz = src.size();
    dst.create(sz, CV_8UC1);

#ifdef HAVE_SSE
    // __m128i ssse3_blue_indices_0  = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12,  9,  6,  3,  0);
    // __m128i ssse3_blue_indices_1  = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11,  8,  5,  2, -1, -1, -1, -1, -1, -1);
    // __m128i ssse3_blue_indices_2  = _mm_set_epi8(13, 10,  7,  4,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // __m128i ssse3_green_indices_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10,  7,  4,  1);
    // __m128i ssse3_green_indices_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12,  9,  6,  3,  0, -1, -1, -1, -1, -1);
    // __m128i ssse3_green_indices_2 = _mm_set_epi8(14, 11,  8,  5,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indices_0   = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11,  8,  5,  2);
    __m128i ssse3_red_indices_1   = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10,  7,  4,  1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indices_2   = _mm_set_epi8(15, 12,  9,  6,  3,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
#endif

    for (int y = 0; y < sz.height; y++)
    {
        const uchar *psrc = src.ptr<uchar>(y);
        uchar *pdst = dst.ptr<uchar>(y);

        int x = 0;

#ifdef HAVE_SSE
        // Here is 16 times unrolled loop for vector processing
        for (; x <= sz.width - 16; x += 16)
        {
            __m128i chunk0 = _mm_loadu_si128((const __m128i*)(psrc + x*3 + 16*0));
            __m128i chunk1 = _mm_loadu_si128((const __m128i*)(psrc + x*3 + 16*1));
            __m128i chunk2 = _mm_loadu_si128((const __m128i*)(psrc + x*3 + 16*2));

            __m128i red = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_red_indices_0),
                                                    _mm_shuffle_epi8(chunk1, ssse3_red_indices_1)),
                                                    _mm_shuffle_epi8(chunk2, ssse3_red_indices_2));

            /* ??? */

            _mm_storeu_si128((__m128i*)(pdst + x), red);
        }
#endif

        // Process leftover pixels
        for (; x < sz.width; x++)
        {
            /* ??? */
        }
    }

    // ! Remove this before writing your optimizations !
    ConvertColor_BGR2GRAY_BT709_fpt(src, dst);
}
