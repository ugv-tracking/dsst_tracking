cmake_minimum_required(VERSION 2.8.7)
project(CfTracking)
ADD_DEFINITIONS ( -std=c++11 )

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include_directories(SYSTEM ${CONAN_INCLUDE_DIRS})
include(CheckCXXCompilerFlag)

find_package(PythonLibs REQUIRED)
find_package(OpenCV REQUIRED)



# add 3rdparty source
set(CF_TCLAP_DIR "src/3rdparty/tclap")
set(CF_CV_EXT_DIR "src/3rdparty/cv_ext")
set(CF_PIOTR_DIR "src/3rdparty/piotr")

include_directories(src/cv_bind
                    src/cf_libs/kcf
                    src/cf_libs/dsst
                    src/cf_libs/common)


include_directories(
    ${PYTHON_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/pybind11/include
    ${OpenCV_DIR}/include
    ${CF_TCLAP_DIR}
    ${CF_CV_EXT_DIR}
    ${CF_PIOTR_DIR}
    ${CF_PIOTR_DIR}/src
    src/main
    src/cf_libs/kcf
    src/cf_libs/dsst
    src/cf_libs/common)

set(CF_PIOTR_SOURCES
    ${CF_PIOTR_DIR}/gradientMex.hpp
    ${CF_PIOTR_DIR}/src/gradientMex.cpp
    ${CF_PIOTR_DIR}/src/sse.hpp
    ${CF_PIOTR_DIR}/src/wrappers.hpp)

set(CF_LIB_COMMON_SOURCES
    src/cf_libs/common/feature_channels.hpp
    src/cf_libs/common/mat_consts.hpp
    src/cf_libs/common/math_helper.hpp
    src/cf_libs/common/math_helper.cpp
    src/cf_libs/common/cf_tracker.hpp
    src/cf_libs/common/tracker_debug.hpp
    src/cf_libs/common/scale_estimator.hpp
    src/cf_libs/common/cv_ext.hpp
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/shift.hpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.hpp
    ${CF_CV_EXT_DIR}/psr.hpp
    ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
    src/main/image_acquisition.hpp
    src/main/image_acquisition.cpp
    ${CF_CV_EXT_DIR}/init_box_selector.hpp
    ${CF_CV_EXT_DIR}/init_box_selector.cpp
    ${CF_CV_EXT_DIR}/tracker_run.hpp
    ${CF_CV_EXT_DIR}/tracker_run.cpp
    ${CF_TCLAP_DIR}/tclap/CmdLine.h)

add_library(cv_bind  SHARED
    src/main/cv_bind.cpp
    src/cv_bind/cv_bind2.h)
target_link_libraries(cv_bind ${OpenCV_LIBS})
set_target_properties(cv_bind PROPERTIES PREFIX "")
set_target_properties(cv_bind PROPERTIES SUFFIX ".so")

add_custom_command(TARGET cv_bind POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/cv_bind.so)

add_library(DSST  SHARED
    src/main/main_dsst.cpp
    src/cf_libs/dsst/dsst_debug.hpp
    src/cf_libs/dsst/dsst_tracker.hpp
    ${CF_MAIN_SOURCES}
    ${CF_LIB_COMMON_SOURCES})
target_link_libraries(DSST ${OpenCV_LIBS})
set_target_properties(DSST PROPERTIES PREFIX "")
set_target_properties(DSST PROPERTIES SUFFIX ".so")

add_custom_command(TARGET DSST POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/DSST.so)
