# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aj/Documents/CV_Coursework/CV_Coursework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aj/Documents/CV_Coursework/CV_Coursework

# Include any dependencies generated for this target.
include CMakeFiles/HybridImage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/HybridImage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HybridImage.dir/flags.make

CMakeFiles/HybridImage.dir/hybrid_image.cpp.o: CMakeFiles/HybridImage.dir/flags.make
CMakeFiles/HybridImage.dir/hybrid_image.cpp.o: hybrid_image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aj/Documents/CV_Coursework/CV_Coursework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/HybridImage.dir/hybrid_image.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HybridImage.dir/hybrid_image.cpp.o -c /home/aj/Documents/CV_Coursework/CV_Coursework/hybrid_image.cpp

CMakeFiles/HybridImage.dir/hybrid_image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HybridImage.dir/hybrid_image.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aj/Documents/CV_Coursework/CV_Coursework/hybrid_image.cpp > CMakeFiles/HybridImage.dir/hybrid_image.cpp.i

CMakeFiles/HybridImage.dir/hybrid_image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HybridImage.dir/hybrid_image.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aj/Documents/CV_Coursework/CV_Coursework/hybrid_image.cpp -o CMakeFiles/HybridImage.dir/hybrid_image.cpp.s

CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.requires:

.PHONY : CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.requires

CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.provides: CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.requires
	$(MAKE) -f CMakeFiles/HybridImage.dir/build.make CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.provides.build
.PHONY : CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.provides

CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.provides.build: CMakeFiles/HybridImage.dir/hybrid_image.cpp.o


# Object files for target HybridImage
HybridImage_OBJECTS = \
"CMakeFiles/HybridImage.dir/hybrid_image.cpp.o"

# External object files for target HybridImage
HybridImage_EXTERNAL_OBJECTS =

HybridImage: CMakeFiles/HybridImage.dir/hybrid_image.cpp.o
HybridImage: CMakeFiles/HybridImage.dir/build.make
HybridImage: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
HybridImage: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
HybridImage: CMakeFiles/HybridImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aj/Documents/CV_Coursework/CV_Coursework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable HybridImage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HybridImage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HybridImage.dir/build: HybridImage

.PHONY : CMakeFiles/HybridImage.dir/build

CMakeFiles/HybridImage.dir/requires: CMakeFiles/HybridImage.dir/hybrid_image.cpp.o.requires

.PHONY : CMakeFiles/HybridImage.dir/requires

CMakeFiles/HybridImage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HybridImage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HybridImage.dir/clean

CMakeFiles/HybridImage.dir/depend:
	cd /home/aj/Documents/CV_Coursework/CV_Coursework && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aj/Documents/CV_Coursework/CV_Coursework /home/aj/Documents/CV_Coursework/CV_Coursework /home/aj/Documents/CV_Coursework/CV_Coursework /home/aj/Documents/CV_Coursework/CV_Coursework /home/aj/Documents/CV_Coursework/CV_Coursework/CMakeFiles/HybridImage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/HybridImage.dir/depend
