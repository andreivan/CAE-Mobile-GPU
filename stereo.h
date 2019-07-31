//
// Created by user on 12/19/2017.
//

#ifndef IMLABOPENCL_STEREO_H
#define IMLABOPENCL_STEREO_H
#ifdef __cplusplus

#include "constants.h"
#include <cassert>
#include <android/log.h>
#include <jni.h>
#include <cstdint>

#define app_name "OpenCLProcessor"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, app_name, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, app_name, __VA_ARGS__))


extern "C" {
#endif

bool compileOpenCLKernels(JNIEnv *env);

double processLightField(JNIEnv *env, jobject instance, jobject imageResult_, jobject imageInput_, int width, int height);

double processLightField_Refocus(JNIEnv *env, jobject instance, jobject imageResult_, jobject imageInput_, int width, int height, int index);
#ifdef __cplusplus
}
#endif

#endif //IMLABOPENCL_STEREO_H
