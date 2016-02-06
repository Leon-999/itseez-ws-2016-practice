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

//#define int_t unsigned int
#define int_t ushort

void ConvertColor_BGR2GRAY_BT709_fpt(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC3 == src.type());
    cv::Size sz = src.size();
    dst.create(sz, CV_8UC1);

    //int shift = 16;
    int shift = 8;

    int_t redC = (int_t)(0.2126f * (1 << shift) + 0.5f);
    int_t greenC = (int_t)(0.7152f * (1 << shift) + 0.5f);
    int_t blueC = (int_t)(0.0722f * (1 << shift) + 0.5f);

    int bias  = 0;

    for (int y = 0; y < sz.height; y++)
    {
        const cv::Vec3b *psrc = src.ptr<cv::Vec3b>(y);
        uchar *pdst = dst.ptr<uchar>(y);

        for (int x = 0; x < sz.width; x++)
        {
            pdst[x] = (redC * psrc[x][2] + greenC * psrc[x][1] + blueC * psrc[x][0] + (1<<(shift-1)) + bias) >> shift;
        }
    }
}

void ConvertColor_BGR2GRAY_BT709_simd(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(CV_8UC3 == src.type());
    cv::Size sz = src.size();
    dst.create(sz, CV_8UC1);

#ifdef HAVE_SSE
    __m128i ssse3_blue_indices_0  = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12,  9,  6,  3,  0);
    __m128i ssse3_blue_indices_1  = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11,  8,  5,  2, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_blue_indices_2  = _mm_set_epi8(13, 10,  7,  4,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indices_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10,  7,  4,  1);
    __m128i ssse3_green_indices_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12,  9,  6,  3,  0, -1, -1, -1, -1, -1);
    __m128i ssse3_green_indices_2 = _mm_set_epi8(14, 11,  8,  5,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indices_0   = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11,  8,  5,  2);
    __m128i ssse3_red_indices_1   = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10,  7,  4,  1, -1, -1, -1, -1, -1);
    __m128i ssse3_red_indices_2   = _mm_set_epi8(15, 12,  9,  6,  3,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    __m128i redCoeff   = _mm_set1_epi16(54);
    __m128i greenCoeff = _mm_set1_epi16(183);
    __m128i blueCoeff  = _mm_set1_epi16(19); // 19 instead of 18 because the sum of 3 coeffs must be == 256
    __m128i bias = _mm_set1_epi16(128);
    __m128i zero = _mm_setzero_si128();
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

            __m128i green = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_green_indices_0),
                                                    _mm_shuffle_epi8(chunk1, ssse3_green_indices_1)),
                                                    _mm_shuffle_epi8(chunk2, ssse3_green_indices_2));

            __m128i blue = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_blue_indices_0),
                                                    _mm_shuffle_epi8(chunk1, ssse3_blue_indices_1)),
                                                    _mm_shuffle_epi8(chunk2, ssse3_blue_indices_2));

            __m128i redLow = _mm_unpacklo_epi8(red, zero);
            __m128i redHigh = _mm_unpackhi_epi8(red, zero);

            __m128i redMultiLow = _mm_mullo_epi16(redLow, redCoeff);
            __m128i redMultiHigh = _mm_mullo_epi16(redHigh, redCoeff);

            __m128i sumLow = _mm_setzero_si128();
            __m128i sumHigh = _mm_setzero_si128();

            sumLow = _mm_add_epi16(sumLow, redMultiLow);
            sumHigh = _mm_add_epi16(sumHigh, redMultiHigh);

            __m128i greenLow = _mm_unpacklo_epi8(green, zero);
            __m128i greenHigh = _mm_unpackhi_epi8(green, zero);

            __m128i greenMultiLow = _mm_mullo_epi16(greenLow, greenCoeff);
            __m128i greenMultiHigh = _mm_mullo_epi16(greenHigh, greenCoeff);

            sumLow = _mm_add_epi16(sumLow, greenMultiLow);
            sumHigh = _mm_add_epi16(sumHigh, greenMultiHigh);

            __m128i blueLow = _mm_unpacklo_epi8(blue, zero);
            __m128i blueHigh = _mm_unpackhi_epi8(blue, zero);

            __m128i blueMultiLow = _mm_mullo_epi16(blueLow, blueCoeff);
            __m128i blueMultiHigh = _mm_mullo_epi16(blueHigh, blueCoeff);

            sumLow = _mm_add_epi16(sumLow, blueMultiLow);
            sumHigh = _mm_add_epi16(sumHigh, blueMultiHigh);

            sumLow = _mm_add_epi16(sumLow, bias);
            sumHigh = _mm_add_epi16(sumHigh, bias);

            int shift = 8;
            sumLow = _mm_srli_epi16(sumLow, shift);
            sumHigh = _mm_srli_epi16(sumHigh, shift);


            __m128i gray_packed = _mm_packus_epi16(sumLow, sumHigh);

            _mm_storeu_si128((__m128i*)(pdst + x), gray_packed);
        }
#endif
        // Process leftover pixels
        for (; x < sz.width; x++)
        {
            float color = 0.2126 * psrc[3 * x + 2] + 0.7152 * psrc[3 * x + 1] + 0.0722 * psrc[3 * x];
            pdst[x] = (int)(color + 0.5);
        }
    }
}
