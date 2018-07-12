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
    int x, y, height, width;
};


class DsstTrackerRun : public TrackerRun
{
public:
    Parameters param;
    TargetFound tFound;

    void setBbox(int x, int y, int height, int width)
    {
        tFound.x      = x;
        tFound.y      = y;
        tFound.height = height;
        tFound.width  = width;
        return;
    }

    void reinit(cv::Mat &img)
   {
        _boundingBox.x      = tFound.x;
        _boundingBox.y      = tFound.y;
        _boundingBox.height = tFound.height;
        _boundingBox.width  = tFound.width;

        _image              = img;
        _targetOnFrame      = _tracker->reinit(_image, _boundingBox);
    }

    void update(cv::Mat &img)
    {
        //STEP1 import new image
        _image = img;
        _targetOnFrame      = _tracker->update(_image, _boundingBox);
        tFound.found        = _targetOnFrame;
        tFound.x            = _boundingBox.x;
        tFound.y            = _boundingBox.y;
        tFound.height       = _boundingBox.height;
        tFound.width        = _boundingBox.width;

        static int frame = 0;
        std::cout << "===========================================" << std::endl;
        if(tFound.found)
            std::cout << "Target found in frame " << frame <<  std::endl;
        else
            std::cout << "Target not found in frame " << frame <<  std::endl;
        std::cout << "x " << tFound.x << " y " << tFound.y << " height " << tFound.height << " width " << tFound.width << std::endl;
        std::cout << "===========================================" << std::endl << std::endl;

        frame++;
        return;
    }

    DsstTrackerRun() : TrackerRun("DSSTcpp")
    {}

    virtual ~DsstTrackerRun()
    {}

    void setParam(double padding)
    {
        cf_tracking::DsstParameters tracker;
        //! set Paras for cf_tracking
        // use original paper parameters from
        // Danelljan, Martin, et al., "Accurate scale estimation for robust visual tracking," in Proc. BMVC, 2014
        {

            tracker.padding = static_cast<double>(padding);
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
            param.sequencePath = "video_path";
            _paras = param;
        }

        return;
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
            .def("setBbox", &DsstTrackerRun::setBbox);

    py::class_<Tracker> (m, "Tracker", dsst_class)
            .def(py::init<>())
            .def("update", &Tracker::update, "A function that update read image from Tusimple",
                 py::arg("image"))
            .def("reinit", &Tracker::reinit, "A function that reinit read image from Tusimple",
                 py::arg("image"));

    return m.ptr();
}
