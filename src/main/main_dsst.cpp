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

namespace py = pybind11;

class DsstTrackerRun : public TrackerRun
{
public:
    Parameters param;
    void play()
    {
        start();
        return;
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

        return;
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
            .def("setParam", &DsstTrackerRun::setParam);

    py::class_<Tracker> (m, "Tracker", dsst_class)
            .def(py::init<>())
            .def("start", &Tracker::start)
            .def("play", &Tracker::play)
            .def_readwrite("param", &Tracker::_paras);

    return m.ptr();
}
