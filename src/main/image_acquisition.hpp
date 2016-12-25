#ifndef IMAGE_ACQUISITION_HPP_
#define IMAGE_ACQUISITION_HPP_

#include "opencv2/highgui/highgui.hpp"

struct ImgAcqParas
{
    bool isMock = false;
    int device = -1;
    std::string sequencePath = "";
    std::string expansionStr = "";
};

class VideoCaptureMock
{
public:
    VideoCaptureMock();

    virtual ~VideoCaptureMock();

    void open();

    VideoCaptureMock& operator >> (CV_OUT cv::Mat& image);

    bool isOpened();

    void release();

private:
    bool isOpen;
    cv::Mat _staticImage;
};

class ImageAcquisition
{
public:
    ImageAcquisition();

    virtual ~ImageAcquisition();

    void open(ImgAcqParas paras);

    void set(int key, int value);

    double get(int key);

    bool isOpened();

    void release();

    ImageAcquisition& operator >> (CV_OUT cv::Mat& image);

private:
    cv::VideoCapture _cvCap;
    ImgAcqParas _paras;
    VideoCaptureMock _mockCap;
};

#endif
