#include "processor.h"
#include "stereo.h"
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
//Camera info structure data containing "calibrated" camera parameter such as: focal length, principal point, etc.
CameraInfo cameraInfo;

bool throwJavaException(JNIEnv *env,std::string method_name,std::string exception_msg, int errorCode=0)
{
    char buf[8];
    sprintf(buf,"%d",errorCode);
    std::string code(buf);

    std::string msg = "@" + method_name + ": " + exception_msg + " ";
    if(errorCode!=0) msg += code;

    jclass generalExp = env->FindClass("java/lang/Exception");
    if (generalExp != 0) {
        env->ThrowNew(generalExp, msg.c_str());
        return true;
    }
    return false;
}

JNIEXPORT jboolean JNICALL Java_com_inha_imlab_opencl_LiveFeatureActivity_compileKernels(JNIEnv *env, jclass clazz)
{
    return compileOpenCLKernels(env);
}


JNIEXPORT double JNICALL
Java_com_inha_imlab_opencl_StaticImageActivity_getDepthLightFieldOpenCL(JNIEnv *env, jobject instance,
                                                                   jobject imageResult_,
                                                                   jobject imageInput,
                                                                   jint width, jint height, jint index) {
    return processLightField(env, instance, imageResult_, imageInput, width, height);
    //return processLightField_Refocus(env, instance, imageResult_, imageInput, width, height, index);
}



void isImageValid(JNIEnv *env, jobject *imageResult_, jobject *imageL_,
                  jobject *imageR_) {
    AndroidBitmapInfo bitmapInfo;
    if(AndroidBitmap_getInfo(env, *imageL_, &bitmapInfo) < 0) {
        throwJavaException(env, "isImageValid", "Error retrieving bitmap metadata from imageL");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaException(env, "isImageValid", "The format of imageL is not RGBA_8888");
    }
    if(AndroidBitmap_getInfo(env, *imageR_, &bitmapInfo) < 0) {
        throwJavaException(env, "isImageValid", "Error retrieving bitmap metadata from imageR");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaException(env, "isImageValid", "The format of imageR is not RGBA_8888");
    }
    if(AndroidBitmap_getInfo(env, *imageResult_, &bitmapInfo) < 0) {
        throwJavaException(env, "isImageValid", "Error retrieving bitmap metadata from imageResult");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaException(env, "isImageValid", "The format of imageResult is not RGBA_8888");
    }
}
