//
// Created by user on 12/19/2017.
//
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "stereo.h"
#include <CL/cl.hpp>
#include <cmath>
#include <android/bitmap.h>
#include <chrono>

static cl::Context      gContext;
static cl::CommandQueue gQueue;
static cl::Kernel       gNV21Kernel;

//Light field
static cl::Kernel       gRemap;
static cl::Kernel       gRemap_Lytro;
static cl::Kernel       gRemap_Image;
static cl::Kernel       gCAE;
static cl::Kernel       gCAE_Native;
static cl::Kernel       gCAE_Initial;
static cl::Kernel       gCAE_Bin;
static cl::Kernel       gWTA_LF;
static cl::Kernel       gSSD;
static cl::Kernel       gRefocus;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// //Utility function
char *file_contents(const char *filename, int *length)
{
    FILE *f = fopen(filename, "r");
    void *buffer;

    if (!f) {
        LOGE("Unable to open %s for reading\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);


    buffer = malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    ((char*)buffer)[*length] = '\0';

    return (char*)buffer;
}

void cb(cl_program p,void* data)
{
    clRetainProgram(p);
    cl_device_id devid[1];
    clGetProgramInfo(p,CL_PROGRAM_DEVICES,sizeof(cl_device_id),(void*)devid,NULL);
    char bug[65536];
    clGetProgramBuildInfo(p,devid[0],CL_PROGRAM_BUILD_LOG,65536*sizeof(char),bug,NULL);
    clReleaseProgram(p);
    LOGE("Build log \n %s\n",bug);
}

//Function name must be different between each cpp file
bool throwJavaExceptions(JNIEnv *env,std::string method_name,std::string exception_msg, int errorCode=0)
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

void checkImage(JNIEnv *env, jobject *imageResult_, jobject *imageL_,
                  jobject *imageR_) {
    AndroidBitmapInfo bitmapInfo;
    if(AndroidBitmap_getInfo(env, *imageL_, &bitmapInfo) < 0) {
        throwJavaExceptions(env, "isImageValid", "Error retrieving bitmap metadata from imageL");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaExceptions(env, "isImageValid", "The format of imageL is not RGBA_8888");
    }
    if(AndroidBitmap_getInfo(env, *imageR_, &bitmapInfo) < 0) {
        throwJavaExceptions(env, "isImageValid", "Error retrieving bitmap metadata from imageR");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaExceptions(env, "isImageValid", "The format of imageR is not RGBA_8888");
    }
    if(AndroidBitmap_getInfo(env, *imageResult_, &bitmapInfo) < 0) {
        throwJavaExceptions(env, "isImageValid", "Error retrieving bitmap metadata from imageResult");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaExceptions(env, "isImageValid", "The format of imageResult is not RGBA_8888");
    }
}

void checkImageLF(JNIEnv *env, jobject *imageResult_, jobject *imageLF_) {
    AndroidBitmapInfo bitmapInfo;
    if(AndroidBitmap_getInfo(env, *imageLF_, &bitmapInfo) < 0) {
        throwJavaExceptions(env, "isLFValid", "Error retrieving bitmap metadata from imageL");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaExceptions(env, "isLFValid", "The format of imageL is not RGBA_8888");
    }
    if(AndroidBitmap_getInfo(env, *imageResult_, &bitmapInfo) < 0) {
        throwJavaExceptions(env, "isLFValid", "Error retrieving bitmap metadata from imageResult");
    }
    if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwJavaExceptions(env, "isLFValid", "The format of imageResult is not RGBA_8888");
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Compile OpenCLKernel initialization
bool compileOpenCLKernels(JNIEnv *env)
{
// Find OCL devices and compile kernels
 cl_int err = CL_SUCCESS;
 try {
     std::vector<cl::Platform> platforms;
     cl::Platform::get(&platforms);
     if (platforms.size() == 0) {
         return false;
     }
     cl_context_properties properties[] =
     { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
     gContext = cl::Context(CL_DEVICE_TYPE_GPU, properties);
     std::vector<cl::Device> devices = gContext.getInfo<CL_CONTEXT_DEVICES>();
     gQueue = cl::CommandQueue(gContext, devices[0], 0, &err);
     int src_length = 0;

     const char* src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/essential.cl",&src_length);
     cl::Program::Sources sources(1,std::make_pair(src, src_length) );
     cl::Program program(gContext, sources);
     program.build(devices,NULL,cb);
     while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gNV21Kernel = cl::Kernel(program, "nv21torgba", &err);
     gRectify = cl::Kernel(program, "rectify", &err);
     gRectify2 = cl::Kernel(program, "rectify2", &err);
     gGaussian = cl::Kernel(program, "gaussianFilter", &err);
     gSobel = cl::Kernel(program, "sobelFilter", &err);
     gGaussianImage = cl::Kernel(program, "gaussianFilterImage", &err);
     gSobelImage = cl::Kernel(program, "sobelFilterImage", &err);
     //gCensusImage = cl::Kernel(program, "censusFilterImage", &err);
     gCensus = cl::Kernel(program, "censusFilter", &err);
     //gWeightMediaFilter = cl::Kernel(program, "medianFilter", &err);
     //gWeightMediaFilter2 = cl::Kernel(program, "weightedMedianFilter2", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/sad_disp.cl",&src_length);
     cl::Program::Sources sourceSAD(1, std::make_pair(src, src_length));
     cl::Program programSAD(gContext, sourceSAD);
     programSAD.build(devices, NULL, cb);
     while(programSAD.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gSADKernelUint32 = cl::Kernel(programSAD, "SAD_Disparity", &err);
     //gSADCensusUint32 = cl::Kernel(programSAD, "SAD_Census", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/asw_disp.cl",&src_length);
     cl::Program::Sources sourceAWS(1, std::make_pair(src, src_length));
     cl::Program programAWS(gContext, sourceAWS);
     programAWS.build(devices, NULL, cb);
     while(programAWS.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gAWSKernel= cl::Kernel(programAWS, "ASW_Disparity", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/cost_volume.cl",&src_length);
     cl::Program::Sources sourceMincost(1, std::make_pair(src, src_length));
     cl::Program programMincost(gContext, sourceMincost);
     programMincost.build(devices, NULL, cb);
     while(programMincost.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gWTA = cl::Kernel(programMincost, "winnerTakesAll_float", &err);
     gWTA2 = cl::Kernel(programMincost, "winnerTakesAll_char", &err);
     gWTA_LF = cl::Kernel(programMincost, "winnerTakesAll_LF", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/weighted_median_filter.cl",&src_length);
     cl::Program::Sources sourceCost(1, std::make_pair(src, src_length));
     cl::Program programCost(gContext, sourceCost);
     programCost.build(devices, NULL, cb);
     while(programCost.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gNewCost = cl::Kernel(programCost, "generateCostVolume", &err);
     gWMF = cl::Kernel(programCost, "weightedMedianFilter", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/cross_disp.cl",&src_length);
     cl::Program::Sources sourceNCC(1, std::make_pair(src, src_length));
     cl::Program programNCC(gContext, sourceNCC);
     programNCC.build(devices, NULL, cb);
     while(programNCC.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     //gNCCKernel = cl::Kernel(programNCC, "NCC_Disparity", &err);
     free((void *)src);
     src = NULL;

     src  = file_contents("/data/data/com.inha.imlab.opencl/app_execdir/constrained_adaptive_entropy.cl",&src_length);
     cl::Program::Sources sourceCAE(1, std::make_pair(src, src_length));
     cl::Program programCAE(gContext, sourceCAE);
     programCAE.build(devices, NULL, cb);
     while(programCAE.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
     gRemap = cl::Kernel(programCAE, "LF_Remap", &err);
     gRemap_Lytro = cl::Kernel(programCAE, "LF_Remap_Lytro", &err);
     gRemap_Image = cl::Kernel(programCAE, "LF_Remap_Image", &err);
     gCAE = cl::Kernel(programCAE, "LF_CAE", &err);
     gCAE_Native = cl::Kernel(programCAE, "LF_CAE_Naive", &err);
     gCAE_Initial = cl::Kernel(programCAE, "LF_CAE_Initial", &err);
     gCAE_Bin = cl::Kernel(programCAE, "LF_CAE_Bin", &err);
     gSSD = cl::Kernel(programCAE, "LF_SSD", &err);
     gRefocus = cl::Kernel(programCAE, "LF_Refocus", &err);
     free((void *)src);
     src = NULL;

     return true;
 }
 catch (cl::Error e) {
     if( !throwJavaExceptions(env,"decode",e.what(),e.err()) )
         LOGI("@decode: %s \n",e.what());
     return false;
 }
}


//Light field
double processLightField(JNIEnv *env, jobject instance, jobject imageResult_, jobject imageInput_, int width, int height) {

    checkImageLF(env, &imageResult_, &imageInput_);

    uint8_t * bitmapLF;
    if (AndroidBitmap_lockPixels(env, imageInput_, (void**)&bitmapLF) < 0) {
        throwJavaExceptions(env, "getDepthImage_", "Failed to lock bitmap pixels of \"bitmapLF\"");
    }
    uint16_t * bitmapResult;
    if (AndroidBitmap_lockPixels(env, imageResult_, (void**)&bitmapResult) < 0) {
        throwJavaExceptions(env, "getDepthImage_", "Failed to lock bitmap pixels of \"bitmapResult\"");
    }

    //Light field parameter
    int depth_resolution = 75;
    float delta =  0.02f;   //0.0214 MONA || 0.0316 PAP || 0.0324 B2 || 0.0416 B1 || 0.0732 LIFE
    // 0.0518 Medieval //
    int UV_diameter = 5;
    int UV_radius = 2;
    int w_spatial = width / UV_diameter;
    int h_spatial = height / UV_diameter;
    int totalPixels = width * height;
    int totalPixels_spatial = w_spatial * h_spatial;
    float sigma = 10.0f;
    int scale = 256 / (depth_resolution);
    float alpha;
    static const cl::ImageFormat format = {CL_RGBA, CL_UNSIGNED_INT8};
    LOGI("width: %d || height: %d || depth search %d", width, height, depth_resolution);

    //Start timer
    std::chrono::high_resolution_clock ::time_point begin = std::chrono::high_resolution_clock::now();
    cl::Image2D imageLF = cl::Image2D(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, format, width, height, 0, bitmapLF);
    //cl::Buffer bufferInLF = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * sizeof(uint32_t), bitmapLF, NULL);
    cl::Buffer bufferRemap = cl::Buffer(gContext, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_float4));
    //cl::Image2D imageRemap = cl::Image2D(gContext, CL_MEM_READ_WRITE, format, width, height, 0, bitmapLF);
    cl::Buffer bufferResponse = cl::Buffer(gContext, CL_MEM_READ_WRITE, (totalPixels_spatial)* depth_resolution * sizeof(cl_float));
    cl::Buffer bufferDepth = cl::Buffer(gContext, CL_MEM_READ_WRITE, (totalPixels_spatial) * sizeof(cl_uchar4));

    std::chrono::high_resolution_clock ::time_point beginLoop;
    std::chrono::high_resolution_clock::time_point endLoop;
    std::chrono::duration<double> time_span;
    try {
        for (int index = 1; index <= depth_resolution; index++) {
            beginLoop = std::chrono::high_resolution_clock::now();

            alpha = -((index - (depth_resolution + 1) / 2) * (delta));
            //Shear the LF with alpha value
            gRemap_Lytro.setArg(0, imageLF);
            gRemap_Lytro.setArg(1, bufferRemap);
            gRemap_Lytro.setArg(2, delta);
            gRemap_Lytro.setArg(3, UV_diameter);
            gRemap_Lytro.setArg(4, UV_radius);
            gRemap_Lytro.setArg(5, alpha);
            gQueue.enqueueNDRangeKernel(gRemap_Lytro,
                                        cl::NullRange,
                                        cl::NDRange(w_spatial, h_spatial),
                                        cl::NDRange(16, 16),
                                        NULL,
                                        NULL);
            gQueue.finish();

            //Calculate the response using CAE
            gCAE.setArg(0, bufferRemap);
            gCAE.setArg(1, bufferResponse);
            gCAE.setArg(2, UV_diameter);
            gCAE.setArg(3, sigma);
            gCAE.setArg(4, index);
            gCAE.setArg(5, depth_resolution);
            gQueue.enqueueNDRangeKernel(gCAE,
                                        cl::NullRange,
                                        cl::NDRange(w_spatial, h_spatial),
                                        cl::NDRange(16, 16),
                                        NULL,
                                        NULL);
            gQueue.finish();
            endLoop = std::chrono::high_resolution_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<float>>(endLoop - beginLoop);
            LOGI("Loop#%d || Alpha: %f || Elapsed Time : %f seconds", index, alpha, time_span);

        }
        gWTA_LF.setArg(0, bufferResponse);
        gWTA_LF.setArg(1, bufferDepth);
        gWTA_LF.setArg(2, depth_resolution);
        gWTA_LF.setArg(3, scale);
        gQueue.enqueueNDRangeKernel(gWTA_LF,
                                    cl::NullRange,
                                    cl::NDRange(w_spatial, h_spatial),
                                    cl::NDRange(16, 16),
                                    NULL,
                                    NULL);
        gQueue.finish();
        LOGI("Transfer buffer back from GPU.....");
        gQueue.enqueueReadBuffer(bufferDepth, CL_TRUE, 0, (totalPixels_spatial) * sizeof(cl_uchar4), bitmapResult);
        //gQueue.enqueueReadBuffer(bufferRemap, CL_TRUE, 0, (totalPixels) * sizeof(cl_uchar4), bitmapResult);
        //LOGI("%d %d %d %d", bitmapLF[0], bitmapLF[1], bitmapLF[2], bitmapLF[3]);
    }catch (cl::Error e) {
        LOGI("helper2@oclDecoder: %s %d \n",e.what(),e.err());
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    AndroidBitmap_unlockPixels(env, imageInput_);
    AndroidBitmap_unlockPixels(env, imageResult_);
    LOGI("End of Program..... timer: %f ", time_span);
    return time_span.count();
}

//Light field refocus
double processLightField_Refocus(JNIEnv *env, jobject instance, jobject imageResult_, jobject imageInput_, int width, int height, int index) {

    checkImageLF(env, &imageResult_, &imageInput_);

    uint32_t * bitmapLF;
    if (AndroidBitmap_lockPixels(env, imageInput_, (void**)&bitmapLF) < 0) {
        throwJavaExceptions(env, "getDepthImage_", "Failed to lock bitmap pixels of \"bitmapLF\"");
    }
    uint32_t * bitmapResult;
    if (AndroidBitmap_lockPixels(env, imageResult_, (void**)&bitmapResult) < 0) {
        throwJavaExceptions(env, "getDepthImage_", "Failed to lock bitmap pixels of \"bitmapResult\"");
    }

    //Light field parameter
    int depth_resolution = 75;
    float delta = 0.0214f;
    int UV_diameter = 9;
    int UV_radius = 4;
    int w_spatial = width / UV_diameter;
    int h_spatial = height / UV_diameter;
    int totalPixels = width * height;
    int totalPixels_spatial = w_spatial * h_spatial;

    LOGI("width: %d || height: %d || depth search %d", width, height, depth_resolution);

    //Start timer
    std::chrono::high_resolution_clock ::time_point begin = std::chrono::high_resolution_clock::now();

    cl::Buffer bufferInLF = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * sizeof(uint32_t), bitmapLF, NULL);
    cl::Buffer bufferRemap = cl::Buffer(gContext, CL_MEM_READ_WRITE, (totalPixels_spatial) * sizeof(cl_uchar4));
    //cl::Memory image = cl::Image2D(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, fmt, width * sizeof(uint32_t), height * sizeof(uint32_t), 0, bitmapLF, &err);

    float alpha = -((71 - (depth_resolution + 1) / 2) * (delta) / 2.0f);
    //Shear the LF with alpha value
    gRefocus.setArg(0, bufferInLF);
    gRefocus.setArg(1, bufferRemap);
    gRefocus.setArg(2, w_spatial);
    gRefocus.setArg(3, h_spatial);
    gRefocus.setArg(4, delta);
    gRefocus.setArg(5, UV_diameter);
    gRefocus.setArg(6, UV_radius);
    gRefocus.setArg(7, alpha);
    gQueue.enqueueNDRangeKernel(gRefocus,
                                cl::NullRange,
                                cl::NDRange(w_spatial, h_spatial),
                                cl::NullRange,
                                NULL,
                                NULL);
    gQueue.finish();
    gQueue.enqueueReadBuffer(bufferRemap, CL_TRUE, 0, (totalPixels_spatial) * sizeof(cl_uchar4), bitmapResult);


    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    AndroidBitmap_unlockPixels(env, imageInput_);
    AndroidBitmap_unlockPixels(env, imageResult_);
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    return time_span.count();

}
