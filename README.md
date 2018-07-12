# Python Warp for DSST tracking

## Build
```
mkdir build
cd build
cmake ..
```

## Testing
```
import sys
sys.path.append("build")
import DSST

padding = 2.5

dsst = DSST.Tracker()
dsst.setParam(padding)
```

## Detail
For other params, please refer to the funciton setParam in file main_dsst.cpp, and change it as your wish.
Currnt parameter setting:
```
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
    param.sequencePath = "/home/i/code_base/ACC_CAR/dsst_tracking/test.avi";
    _paras = param;
}

```

