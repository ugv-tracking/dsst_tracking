#include "image_acquisition.hpp"

ImageAcquisition::ImageAcquisition()
{
}

ImageAcquisition& ImageAcquisition::operator>>(CV_OUT cv::Mat& image)
{
    if (_paras.isMock)
        _mockCap >> image;
    else
        _cvCap >> image;

    return *this;
}

void ImageAcquisition::release()
{
    if (!_paras.isMock)
        _cvCap.release();
}

bool ImageAcquisition::isOpened()
{
    if (_paras.isMock)
        return _mockCap.isOpened();
    else
        return _cvCap.isOpened();
}

void ImageAcquisition::set(int key, int value)
{
    if (!_paras.isMock)
        _cvCap.set(key, value);
}

void ImageAcquisition::open(ImgAcqParas paras)
{
    _paras = paras;

    if (_paras.isMock)
    {
        _mockCap.open();
    }
    else
    {
        if (_paras.sequencePath.empty())
            _cvCap.open(_paras.device);
        else
        {
            std::string sequenceExpansion =
                _paras.sequencePath + _paras.expansionStr;

            _cvCap.open(sequenceExpansion);
        }
    }
}

ImageAcquisition::~ImageAcquisition()
{
}

double ImageAcquisition::get(int key)
{
    if (!_paras.isMock)
        return _cvCap.get(key);

    return 0.0;
}

void VideoCaptureMock::release()
{
}

bool VideoCaptureMock::isOpened()
{
    return isOpen;
}

VideoCaptureMock& VideoCaptureMock::operator>>(CV_OUT cv::Mat& image)
{
    image = _staticImage;
    return *this;
}

void VideoCaptureMock::open()
{
    isOpen = true;
}

VideoCaptureMock::~VideoCaptureMock()
{
}

VideoCaptureMock::VideoCaptureMock() : isOpen(false)
{
    _staticImage = cv::Mat(360, 640, CV_8UC3);
    cv::randu(_staticImage, cv::Scalar::all(0), cv::Scalar::all(255));
}
