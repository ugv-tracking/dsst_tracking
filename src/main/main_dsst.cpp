/*
// License Agreement (3-clause BSD License)
// Copyright (c) 2015, Klaus Haag, all rights reserved.
// Third party copyrights and patents are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
*/

#include <pybind11/pybind11.h>
#include <tclap/CmdLine.h>
#include <iostream>
#include "dsst_tracker.hpp"
#include "tracker_run.hpp"
#include "init_box_selector.hpp"
#include "cf_tracker.hpp"
#include "pybind11_opencv_typecaster_v2.h"

namespace py = pybind11;
using namespace cv;

struct TargetFound
{
    bool found;
    double x, y, height, width;
};


class DsstTrackerRun : public TrackerRun
{
public:
    Parameters param;
    TargetFound tFound;

    void play(cv::Mat img)
    {
        //STEP1 import new image
        _image = img;
        std::cout << img.cols << " " << img.rows << std::endl;


        //STEP2 if not initialized, do initialization, else update
        if (!_isTrackerInitialzed)
        {
            if (!_hasInitBox)
            {
                Rect box;
                if (!InitBoxSelector::selectBox(_image, box))
                    return;

                _boundingBox = Rect_<double>(static_cast<double>(box.x),
                    static_cast<double>(box.y),
                    static_cast<double>(box.width),
                    static_cast<double>(box.height));
                _hasInitBox = true;
            }
            _targetOnFrame = _tracker->reinit(_image, _boundingBox);
            if (_targetOnFrame)
                _isTrackerInitialzed = true;
        }
        else
        {
            _isStep = false;
            _targetOnFrame = _tracker->update(_image, _boundingBox);
        }

        //STEP3 output result
        {
            Mat hudImage;
            _image.copyTo(hudImage);
            rectangle(hudImage, _boundingBox, Scalar(0, 0, 255), 2);
            Point_<double> center;
            center.x = _boundingBox.x + _boundingBox.width / 2;
            center.y = _boundingBox.y + _boundingBox.height / 2;
            circle(hudImage, center, 3, Scalar(0, 0, 255), 2);

            if (!_targetOnFrame)
            {
                cv::Point_<double> tl = _boundingBox.tl();
                cv::Point_<double> br = _boundingBox.br();

                line(hudImage, tl, br, Scalar(0, 0, 255));
                line(hudImage, cv::Point_<double>(tl.x, br.y),
                    cv::Point_<double>(br.x, tl.y), Scalar(0, 0, 255));
            }

            imshow(_windowTitle.c_str(), hudImage);

            tFound.found = _targetOnFrame;
            tFound.x = _boundingBox.x;
            tFound.y = _boundingBox.y;
            tFound.height = _boundingBox.height;
            tFound.width  = _boundingBox.width;
        }
    }

    DsstTrackerRun() : TrackerRun("DSSTcpp")
    {}

    virtual ~DsstTrackerRun()
    {}

    void setParam()
    {
        cf_tracking::DsstParameters tracker;
        //! set Paras for cf_tracking
        // use original paper parameters from
        // Danelljan, Martin, et al., "Accurate scale estimation for robust visual tracking," in Proc. BMVC, 2014
        {

            tracker.padding = static_cast<double>(1);
            tracker.outputSigmaFactor = static_cast<double>(1.0 / 16.0);
            tracker.lambda = static_cast<double>(0.01);
            tracker.learningRate = static_cast<double>(0.025);
            tracker.templateSize = 100;
            tracker.cellSize = 1;

            tracker.enableTrackingLossDetection = false;
            tracker.psrThreshold = 0;
            tracker.psrPeakDel = 1;

            tracker.enableScaleEstimator = true;
            tracker.scaleSigmaFactor = static_cast<double>(0.25);
            tracker.scaleStep = static_cast<double>(1.02);
            tracker.scaleCellSize = 4;
            tracker.numberOfScales = 33;

            tracker.originalVersion = true;
            tracker.resizeType = cv::INTER_AREA;
            _tracker = new cf_tracking::DsstTracker(tracker);
        }

        //! set Paras for data play
        {
            param.sequencePath = "/home/i/code_base/ACC_CAR/dsst_tracking/test.avi";
            _paras = param;
        }

        //! set opecv
        _windowTitle = "Dsst Tracking";
        namedWindow(_windowTitle.c_str());

        return;
    }


    void show_image(cv::Mat image)
    {
        cv::imshow("image_from_Cpp", image);
        cv::waitKey(0);
    }

    cv::Mat read_image(std::string image_name)
    {
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
        return image;
    }

private:
    cf_tracking::DsstDebug<cf_tracking::DsstTracker::T> _debug;
};

class Tracker : public DsstTrackerRun {
public:
    using DsstTrackerRun::DsstTrackerRun;

    cf_tracking::CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd) override
    {
        PYBIND11_OVERLOAD_PURE (
            cf_tracking::CfTracker*,
            DsstTrackerRun,
            parseTrackerParas,
            cmd
        );
    }

    cf_tracking::CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv) override
    {
        PYBIND11_OVERLOAD_PURE (
            cf_tracking::CfTracker*,
            DsstTrackerRun,
            parseTrackerParas,
            cmd, argc, **argv
        );
    }
};


PYBIND11_PLUGIN(DSST) {
    py::module m("DSST", "DSST plugin");

    py::class_<TargetFound> targetfound(m, "TargetFound");
    targetfound
            .def_readonly("found", &TargetFound::found)
            .def_readonly("x", &TargetFound::x)
            .def_readonly("y", &TargetFound::y)
            .def_readonly("height", &TargetFound::height)
            .def_readonly("width", &TargetFound::width);

    py::class_<Parameters> params (m, "Parameters");
    params
            .def(py::init<>())
            .def_readwrite("sequencePath", &Parameters::sequencePath)
            .def_readwrite("outputFilePath", &Parameters::outputFilePath)
            .def_readwrite("imgExportPath", &Parameters::imgExportPath)
            .def_readwrite("expansion", &Parameters::expansion)
            .def_readwrite("initBb", &Parameters::initBb)
            .def_readwrite("device", &Parameters::device)
            .def_readwrite("startFrame", &Parameters::startFrame)
            .def_readwrite("showOutput", &Parameters::showOutput)
            .def_readwrite("paused", &Parameters::paused)
            .def_readwrite("repeat", &Parameters::repeat)
            .def_readwrite("isMockSequence", &Parameters::isMockSequence);


    py::class_<DsstTrackerRun> dsst_class(m, "DsstTrackerRun");
    dsst_class
            .def("setParam", &DsstTrackerRun::setParam)
            .def_readwrite("param", &DsstTrackerRun::param)
            .def_readonly("tFound",&DsstTrackerRun::tFound)
            .def("read_image", &DsstTrackerRun::read_image, "A function that read an image",
                py::arg("image"))
            .def("show_image", &DsstTrackerRun::show_image, "A function that show an image",
                py::arg("image"));

    py::class_<Tracker> (m, "Tracker", dsst_class)
            .def(py::init<>())
            .def("play", &Tracker::play, "A function that read image from Tusimple",
                 py::arg("image"));

    return m.ptr();
}
