LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := OpenCLProcessor
LOCAL_ARM_MODE := arm
LOCAL_SRC_FILES := processor.cpp stereo.cpp
LOCAL_CPPFLAGS	:= -DARM -DOS_LNX -DARCH_32 -fexceptions
LOCAL_CPPFLAGS	+= -I$(LOCAL_PATH)/include
LOCAL_CPPFLAGS 	+= -fexceptions -std=c++11
LOCAL_LDLIBS := -ljnigraphics -llog $(LOCAL_PATH)/libs/libOpenCL.so
include $(BUILD_SHARED_LIBRARY)
