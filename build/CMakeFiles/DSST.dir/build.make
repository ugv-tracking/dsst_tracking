# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/i/code_base/ACC_CAR/dsst_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/i/code_base/ACC_CAR/dsst_tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/DSST.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DSST.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DSST.dir/flags.make

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o: ../src/main/main_dsst.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/main/main_dsst.cpp

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/main/main_dsst.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/main/main_dsst.cpp > CMakeFiles/DSST.dir/src/main/main_dsst.cpp.i

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/main/main_dsst.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/main/main_dsst.cpp -o CMakeFiles/DSST.dir/src/main/main_dsst.cpp.s

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.requires

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.provides: CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.provides

CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.provides.build: CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o: ../src/main/image_acquisition.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/main/image_acquisition.cpp

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/main/image_acquisition.cpp > CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.i

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/main/image_acquisition.cpp -o CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.s

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.requires

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.provides: CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.provides

CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.provides.build: CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o: ../src/3rdparty/cv_ext/init_box_selector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/init_box_selector.cpp

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/init_box_selector.cpp > CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.i

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/init_box_selector.cpp -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.s

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.requires

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.provides: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.provides

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.provides.build: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o: ../src/3rdparty/cv_ext/tracker_run.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/tracker_run.cpp

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/tracker_run.cpp > CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.i

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/tracker_run.cpp -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.s

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.requires

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.provides: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.provides

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.provides.build: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o: ../src/cf_libs/common/math_helper.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/cf_libs/common/math_helper.cpp

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/cf_libs/common/math_helper.cpp > CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.i

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/cf_libs/common/math_helper.cpp -o CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.s

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.requires

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.provides: CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.provides

CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.provides.build: CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o: ../src/3rdparty/cv_ext/shift.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/shift.cpp

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/shift.cpp > CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.i

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/shift.cpp -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.s

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.requires

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.provides: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.provides

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.provides.build: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o: ../src/3rdparty/cv_ext/math_spectrums.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/math_spectrums.cpp

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/math_spectrums.cpp > CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.i

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/cv_ext/math_spectrums.cpp -o CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.s

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.requires

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.provides: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.provides

CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.provides.build: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o: CMakeFiles/DSST.dir/flags.make
CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o: ../src/3rdparty/piotr/src/gradientMex.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o -c /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/piotr/src/gradientMex.cpp

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/piotr/src/gradientMex.cpp > CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.i

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/i/code_base/ACC_CAR/dsst_tracking/src/3rdparty/piotr/src/gradientMex.cpp -o CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.s

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.requires:
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.requires

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.provides: CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.requires
	$(MAKE) -f CMakeFiles/DSST.dir/build.make CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.provides.build
.PHONY : CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.provides

CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.provides.build: CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o

# Object files for target DSST
DSST_OBJECTS = \
"CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o" \
"CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o" \
"CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o" \
"CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o" \
"CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o" \
"CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o" \
"CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o" \
"CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o"

# External object files for target DSST
DSST_EXTERNAL_OBJECTS =

DSST.so: CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o
DSST.so: CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o
DSST.so: CMakeFiles/DSST.dir/build.make
DSST.so: /usr/local/lib/libopencv_viz.so.3.1.0
DSST.so: /usr/local/lib/libopencv_videostab.so.3.1.0
DSST.so: /usr/local/lib/libopencv_videoio.so.3.1.0
DSST.so: /usr/local/lib/libopencv_video.so.3.1.0
DSST.so: /usr/local/lib/libopencv_superres.so.3.1.0
DSST.so: /usr/local/lib/libopencv_stitching.so.3.1.0
DSST.so: /usr/local/lib/libopencv_shape.so.3.1.0
DSST.so: /usr/local/lib/libopencv_photo.so.3.1.0
DSST.so: /usr/local/lib/libopencv_objdetect.so.3.1.0
DSST.so: /usr/local/lib/libopencv_ml.so.3.1.0
DSST.so: /usr/local/lib/libopencv_imgproc.so.3.1.0
DSST.so: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
DSST.so: /usr/local/lib/libopencv_highgui.so.3.1.0
DSST.so: /usr/local/lib/libopencv_flann.so.3.1.0
DSST.so: /usr/local/lib/libopencv_features2d.so.3.1.0
DSST.so: /usr/local/lib/libopencv_core.so.3.1.0
DSST.so: /usr/local/lib/libopencv_calib3d.so.3.1.0
DSST.so: /usr/local/lib/libopencv_features2d.so.3.1.0
DSST.so: /usr/local/lib/libopencv_ml.so.3.1.0
DSST.so: /usr/local/lib/libopencv_highgui.so.3.1.0
DSST.so: /usr/local/lib/libopencv_videoio.so.3.1.0
DSST.so: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
DSST.so: /usr/local/lib/libopencv_flann.so.3.1.0
DSST.so: /usr/local/lib/libopencv_video.so.3.1.0
DSST.so: /usr/local/lib/libopencv_imgproc.so.3.1.0
DSST.so: /usr/local/lib/libopencv_core.so.3.1.0
DSST.so: CMakeFiles/DSST.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library DSST.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DSST.dir/link.txt --verbose=$(VERBOSE)
	strip /home/i/code_base/ACC_CAR/dsst_tracking/build/DSST.so

# Rule to build all files generated by this target.
CMakeFiles/DSST.dir/build: DSST.so
.PHONY : CMakeFiles/DSST.dir/build

CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/main/main_dsst.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/main/image_acquisition.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/init_box_selector.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/tracker_run.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/cf_libs/common/math_helper.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/shift.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/3rdparty/cv_ext/math_spectrums.cpp.o.requires
CMakeFiles/DSST.dir/requires: CMakeFiles/DSST.dir/src/3rdparty/piotr/src/gradientMex.cpp.o.requires
.PHONY : CMakeFiles/DSST.dir/requires

CMakeFiles/DSST.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DSST.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DSST.dir/clean

CMakeFiles/DSST.dir/depend:
	cd /home/i/code_base/ACC_CAR/dsst_tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/i/code_base/ACC_CAR/dsst_tracking /home/i/code_base/ACC_CAR/dsst_tracking /home/i/code_base/ACC_CAR/dsst_tracking/build /home/i/code_base/ACC_CAR/dsst_tracking/build /home/i/code_base/ACC_CAR/dsst_tracking/build/CMakeFiles/DSST.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DSST.dir/depend

