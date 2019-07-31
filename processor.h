#ifndef __JNI_H__
#define __JNI_H__
#ifdef __cplusplus

#include <android/bitmap.h>
#include <android/log.h>
#include "constants.h"
#include <chrono>
#include <cassert>

#define app_name "OpenCLProcessor"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, app_name, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, app_name, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, app_name, __VA_ARGS__))

extern "C" {
#endif

#define DECLARE_NOPARAMS(returnType,fullClassName,func) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jclass clazz);
#define DECLARE_NOPARAMS2(returnType,fullClassName,func) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jobject instance);

#define DECLARE_WITHPARAMS(returnType,fullClassName,func,...) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jclass clazz,__VA_ARGS__);
#define DECLARE_WITHPARAMS2(returnType,fullClassName,func,...) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jobject instance,__VA_ARGS__);

DECLARE_NOPARAMS(jboolean,com_inha_imlab_opencl_LiveFeatureActivity,compileKernels)
DECLARE_NOPARAMS2(jboolean,com_inha_imlab_opencl_StaticImageActivity,compileKernels)

DECLARE_WITHPARAMS(double,com_inha_imlab_opencl_CameraPreview,runfilter,jobject outBmp, jbyteArray inData, jint width, jint height, jint choice)
DECLARE_WITHPARAMS(double,com_inha_imlab_opencl_LiveFeatureActivity,runfilter,jobject outBmp, jbyteArray inData, jint width, jint height, jint choice)

DECLARE_WITHPARAMS2(double,com_inha_imlab_opencl_StaticImageActivity,getDepthLightFieldOpenCL,jobject imageResult_, jobject imageInput, jint width, jint height, jint index)

JNIEXPORT double JNICALL
Java_com_inha_imlab_opencl_LiveFeatureActivity_getDepthImage(JNIEnv *env, jobject instance,
                                                             jobject out, jbyteArray in_,
                                                             jint width, jint height,
                                                             jint dispRange, jint dispScale,
                                                             jint mSize, jint algorithm, jint preproc, jint postproc);

void helper(uint32_t* out, int osize, uint8_t* in, int isize, int w, int h, int choice);
void helper2(uint32_t* out, uint8_t* in, int width, int height, int dispRange, int dispScale, int mSize, int algorithm, int preproc, int postproc);

void isImageValid(JNIEnv *env, jobject *imageResult_, jobject *imageL_,
                  jobject *imageR_);

//Global variable
double coeffx = 0.0, coeffy = 0.0;
double xd = 0.0, yd = 0.0;
double x = 0.0, y = 0.0;
int i = 0, j = 0;
int a = 0, b = 0;
double r_sqr = 0.0;

#ifndef ARR_MACRO
#define ARR_MACRO
#define at2(arr, x, y, width) arr[x * width + y]
#define at3(arr, x, y, z, depth, width) arr[x * depth * width + y * depth + z]
#endif

#ifdef __cplusplus
}
#endif

#endif //__JNI_H__
