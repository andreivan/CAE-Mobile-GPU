/*
Constrained Adaptive Entropy (CAE)
*/
#define at3(arr, x, y, z, depth, width) arr[x * depth * width + y * depth + z]

// PADDING Y
int index_y(int y, int height)
	{
	if (0 <= y && y < height)
		return y;
    else if (y < 0)
        return 0;
    else
        return height-1;
	}
// PADDING X
int index_x(int x, int width)
	{
	if (0 <= x && x < width)
		return x;
    else if (x < 0)
        return 0;
    else
        return width-1;
    }
__constant sampler_t sampler =
      CLK_FILTER_NEAREST
    | CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void LF_Remap_Image(
    __read_only image2d_t input,
    __global float4* output,
    float delta,
    int UV_diameter,
    int UV_radius,
    float alpha)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int                 x_1,x_2,y_1,y_2                                 ;
    int                 i,j                                             ;
    float               x_ind,y_ind                                     ;
    float               x_floor,y_floor                                 ;
    float               x_1_w,x_2_w,y_1_w,y_2_w                         ;
    int                 x_1_index,x_2_index,y_1_index,y_2_index         ;
    int                 x_index_remap,y_index_remap                     ;
    float4              interp_color_RGB = 0.0f                         ;
    float4              tempX, tempY, tempZ, tempW                      ;

    int     stereo_diff = UV_radius                             ;
    int     window_size = UV_diameter*UV_diameter               ;
    int     height_of_remap = height*UV_diameter                ;

    for (i = -stereo_diff; i < stereo_diff+1; ++i)
        for (j = -stereo_diff; j < stereo_diff+1; ++j)
        {
            x_ind   = ((float)i)*(alpha) + ((float)x)           ;
            y_ind   = ((float)j)*(-1.0f*alpha) + ((float)y)     ;

            x_floor = floor(x_ind)  ;
            y_floor = floor(y_ind)  ;

            x_1     = index_x(x_floor  ,width )     ;
            y_1     = index_y(y_floor  ,height)     ;
            x_2     = index_x(x_floor+1,width )     ;
            y_2     = index_y(y_floor+1,height)     ;

            x_1_w   = 1 - (x_ind-x_floor)        ;
            x_2_w   = 1 - x_1_w                  ;
            y_1_w   = 1 - (y_ind-y_floor)        ;
            y_2_w   = 1 - y_1_w                  ;

            x_1_index = i+stereo_diff + (x_1)*UV_diameter   ;
            y_1_index = j+stereo_diff + (y_1)*UV_diameter   ;
            x_2_index = i+stereo_diff + (x_2)*UV_diameter   ;
            y_2_index = j+stereo_diff + (y_2)*UV_diameter   ;
            tempX  = convert_float4(read_imageui(input, sampler, (int2)(y_1_index, x_1_index)));
            tempY  = convert_float4(read_imageui(input, sampler, (int2)(y_2_index, x_1_index)));
            tempZ  = convert_float4(read_imageui(input, sampler, (int2)(y_1_index, x_2_index)));
            tempW  = convert_float4(read_imageui(input, sampler, (int2)(y_2_index, x_2_index)));

            // R->X  G->Y  B->Z
            interp_color_RGB = y_1_w*x_1_w*tempX+
                               y_2_w*x_1_w*tempY+
                               y_1_w*x_2_w*tempZ+
                               y_2_w*x_2_w*tempW;

            x_index_remap = i+stereo_diff + (x)*UV_diameter   ;
            y_index_remap = j+stereo_diff + (y)*UV_diameter   ;
            output[y_index_remap + x_index_remap*height_of_remap] = (interp_color_RGB);
            //write_imageui(output, (int2)(y_index_remap, x_index_remap), convert_uint4(interp_color_RGB));
        }
}

__kernel void LF_Remap(
    __global uchar4* input,
    __global float4* output,
    float delta,
    int UV_diameter,
    int UV_radius,
    float alpha)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    unsigned int        x_1,x_2,y_1,y_2                                 ;
    int                 i,j                                             ;
    float               x_ind,y_ind                                     ;
    float               x_floor,y_floor                                 ;
    float               x_1_w,x_2_w,y_1_w,y_2_w                         ;
    unsigned int        x_1_index,x_2_index,y_1_index,y_2_index         ;
    unsigned int        x_index_remap,y_index_remap                     ;
    float4              interp_color_RGB                                ;
    float4              tempX, tempY, tempZ, tempW                      ;

    int     stereo_diff = UV_radius                             ;
    int     window_size = UV_diameter*UV_diameter               ;
    int     height_of_remap = height*UV_diameter                ;

    for (i = -stereo_diff; i < stereo_diff+1; ++i)
        for (j = -stereo_diff; j < stereo_diff+1; ++j)
        {
            x_ind   = ((float)i)*(alpha) + ((float)x)           ;
            y_ind   = ((float)j)*(-1.0f*alpha) + ((float)y)     ;

            x_floor = floor(x_ind)  ;
            y_floor = floor(y_ind)  ;

            x_1     = index_x(x_floor  ,width )     ;
            y_1     = index_y(y_floor  ,height)     ;
            x_2     = index_x(x_floor+1,width )     ;
            y_2     = index_y(y_floor+1,height)     ;

            x_1_w   = 1 - (x_ind-x_floor)        ;
            x_2_w   = 1 - x_1_w                  ;
            y_1_w   = 1 - (y_ind-y_floor)        ;
            y_2_w   = 1 - y_1_w                  ;

            x_1_index = i+stereo_diff + (x_1)*UV_diameter   ;
            y_1_index = j+stereo_diff + (y_1)*UV_diameter   ;
            x_2_index = i+stereo_diff + (x_2)*UV_diameter   ;
            y_2_index = j+stereo_diff + (y_2)*UV_diameter   ;

            tempX = convert_float4(input[y_1_index+x_1_index*height_of_remap]);
            tempY = convert_float4(input[y_2_index+x_1_index*height_of_remap]);
            tempZ = convert_float4(input[y_1_index+x_2_index*height_of_remap]);
            tempW = convert_float4(input[y_2_index+x_2_index*height_of_remap]);
            // R->X  G->Y  B->Z
            interp_color_RGB = y_1_w*x_1_w*tempX+
                                 y_2_w*x_1_w*tempY+
                                 y_1_w*x_2_w*tempZ+
                                 y_2_w*x_2_w*tempW;

            x_index_remap = i+stereo_diff + (x)*UV_diameter   ;
            y_index_remap = j+stereo_diff + (y)*UV_diameter   ;

            output[y_index_remap + x_index_remap*height_of_remap] = (interp_color_RGB);
        }
}

__kernel void LF_Remap_Lytro(
    __read_only image2d_t input,
    __global float4* output,
    float delta,
    int UV_diameter,
    int UV_radius,
    float alpha)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int                 x_1,x_2,y_1,y_2                                 ;
    int                 i,j                                             ;
    float               x_ind,y_ind                                     ;
    float               x_floor,y_floor                                 ;
    float               x_1_w,x_2_w,y_1_w,y_2_w                         ;
    int                 x_1_index,x_2_index,y_1_index,y_2_index         ;
    int                 x_index_remap,y_index_remap                     ;
    float4              interp_color_RGB = 0.0f                         ;
    float4              tempX, tempY, tempZ, tempW                      ;

    int     stereo_diff = UV_radius                             ;
    int     window_size = UV_diameter*UV_diameter               ;
    int     height_of_remap = height*UV_diameter                ;

    for (i = -stereo_diff; i < stereo_diff+1; ++i)
        for (j = -stereo_diff; j < stereo_diff+1; ++j)
        {
            x_ind   = ((float)i)*(-1.0f*alpha) + ((float)x)     ;
            y_ind   = ((float)j)*(-1.0f*alpha) + ((float)y)     ;

            x_floor = floor(x_ind)  ;
            y_floor = floor(y_ind)  ;

            x_1     = index_x(x_floor  ,width )     ;
            y_1     = index_y(y_floor  ,height)     ;
            x_2     = index_x(x_floor+1,width )     ;
            y_2     = index_y(y_floor+1,height)     ;

            x_1_w   = 1 - (x_ind-x_floor)        ;
            x_2_w   = 1 - x_1_w                  ;
            y_1_w   = 1 - (y_ind-y_floor)        ;
            y_2_w   = 1 - y_1_w                  ;

            x_1_index = i+stereo_diff + (x_1)*UV_diameter   ;
            y_1_index = j+stereo_diff + (y_1)*UV_diameter   ;
            x_2_index = i+stereo_diff + (x_2)*UV_diameter   ;
            y_2_index = j+stereo_diff + (y_2)*UV_diameter   ;
            tempX  = convert_float4(read_imageui(input, sampler, (int2)(y_1_index, x_1_index)));
            tempY  = convert_float4(read_imageui(input, sampler, (int2)(y_2_index, x_1_index)));
            tempZ  = convert_float4(read_imageui(input, sampler, (int2)(y_1_index, x_2_index)));
            tempW  = convert_float4(read_imageui(input, sampler, (int2)(y_2_index, x_2_index)));

            // R->X  G->Y  B->Z
            interp_color_RGB = y_1_w*x_1_w*tempX+
                               y_2_w*x_1_w*tempY+
                               y_1_w*x_2_w*tempZ+
                               y_2_w*x_2_w*tempW;

            x_index_remap = i+stereo_diff + (x)*UV_diameter   ;
            y_index_remap = j+stereo_diff + (y)*UV_diameter   ;
            output[y_index_remap + x_index_remap*height_of_remap] = (interp_color_RGB);
            //write_imageui(output, (int2)(y_index_remap, x_index_remap), convert_uint4(interp_color_RGB));
        }
}

__kernel void LF_CAE(
    __global float4* input,
    __global float* response,
    int UV_diameter,
    float sigma,
    int disparity,
    int depth_resolution)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    float4 center_patch             ;

    int                 i,j                 ;
    int                 x_ind,y_ind         ;
    int                 index               ;
    int     window_size     = UV_diameter*UV_diameter           ;
    int     height_of_remap = height*UV_diameter                ;

    float3 histRGB[256]         ;
    float3 sumRGB = 0.0f        ;
    float3 max_valRGB = 0.0f    ;
    float3 correspRGB           ;
    float3 pixelRGB             ;
    float3 cRGB                 ;
    int3   indexRGB             ;
    float3 reducesRGB           ;
    float3 numeratorRGB         ;

    float reduces       ;
    float numerator     ;
    float exponential   ;
    float denominator = 2.0f * sigma * sigma          ;

    x_ind = ((UV_diameter+1)/2) + x * UV_diameter  ;
    y_ind = ((UV_diameter+1)/2) + y * UV_diameter  ;

    center_patch = (input[y_ind + x_ind*height_of_remap]);
    cRGB = center_patch.xyz     ;

    float3 weight;
    float3 floorRGB;
    float3 ceilRGB;
    float3 weightRGB;

    for (j = 0; j < UV_diameter; j++)
    {
        x_ind = j + x * UV_diameter;
        y_ind = y * UV_diameter;
        index = y_ind + x_ind*height_of_remap;
        center_patch = (input[index]);

        //R
        pixelRGB.x = center_patch.x     ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y     ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z     ;
        indexRGB.z = round(pixelRGB.z)  ;

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + (weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + (weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + (weight.z); //Equation (10)
        /////////////////////////////////////////////////////////////////////////////
        center_patch = (input[++index]);

        //R
        pixelRGB.x = center_patch.x     ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y     ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z     ;
        indexRGB.z = round(pixelRGB.z)  ;

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + (weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + (weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + (weight.z); //Equation (10)
        /////////////////////////////////////////////////////////////////////////////
        center_patch = (input[++index]);

        //R
        pixelRGB.x = center_patch.x     ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y     ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z     ;
        indexRGB.z = round(pixelRGB.z)  ;

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + (weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + (weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + (weight.z); //Equation (10)
        /////////////////////////////////////////////////////////////////////////////
        center_patch = (input[++index]);

        //R
        pixelRGB.x = center_patch.x     ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y     ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z     ;
        indexRGB.z = round(pixelRGB.z)  ;

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + (weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + (weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + (weight.z); //Equation (10)
        /////////////////////////////////////////////////////////////////////////////
        center_patch = (input[++index]);

        //R
        pixelRGB.x = center_patch.x     ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y     ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z     ;
        indexRGB.z = round(pixelRGB.z)  ;

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + (weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + (weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + (weight.z); //Equation (10)
    }

    //Equation (12)
    float3 w_s = (float3)window_size;

    float4 hist_1, hist_2, hist_3;
    float4 ones = (float4)1.0f;

    for (i = 0 ; i < 256 ; i+=4)
    {
        histRGB[i] = histRGB[i] / w_s;
        histRGB[i+1] = histRGB[i+1] / w_s;
        histRGB[i+2] = histRGB[i+2] / w_s;
        histRGB[i+3] = histRGB[i+3] / w_s;

        hist_1.x = histRGB[i].x   != 0 ? histRGB[i].x   : 0.0f;
        hist_1.y = histRGB[i+1].x != 0 ? histRGB[i+1].x : 0.0f;
        hist_1.z = histRGB[i+2].x != 0 ? histRGB[i+2].x : 0.0f;
        hist_1.w = histRGB[i+3].x != 0 ? histRGB[i+3].x : 0.0f;

        hist_2.x = histRGB[i].y   != 0 ? histRGB[i].y   : 0.0f;
        hist_2.y = histRGB[i+1].y != 0 ? histRGB[i+1].y : 0.0f;
        hist_2.z = histRGB[i+2].y != 0 ? histRGB[i+2].y : 0.0f;
        hist_2.w = histRGB[i+3].y != 0 ? histRGB[i+3].y : 0.0f;

        hist_3.x = histRGB[i].z   != 0 ? histRGB[i].z   : 0.0f;
        hist_3.y = histRGB[i+1].z != 0 ? histRGB[i+1].z : 0.0f;
        hist_3.z = histRGB[i+2].z != 0 ? histRGB[i+2].z : 0.0f;
        hist_3.w = histRGB[i+3].z != 0 ? histRGB[i+3].z : 0.0f;
        sumRGB.x += dot(hist_1, ones);
        sumRGB.y += dot(hist_2, ones);
        sumRGB.z += dot(hist_3, ones);
    }

    /*
    #pragma unroll
    for(i=0;i<256;i++)
    {
        histRGB[i] = histRGB[i] / w_s;
        if (histRGB[i].x > 0)
            sumRGB.x += histRGB[i].x ;
        if (histRGB[i].y > 0)
            sumRGB.y += histRGB[i].y ;
        if (histRGB[i].z > 0)
            sumRGB.z += histRGB[i].z ;
    }
    */
    #pragma unroll
    for(i=0;i<256;i++)
    {
        //|g| is histRGB[i] / sumRGB
        if(histRGB[i].x > 0)
            max_valRGB.x += (histRGB[i].x / sumRGB.x) * log(histRGB[i].x);
        if(histRGB[i].y > 0)
            max_valRGB.y += (histRGB[i].y / sumRGB.y) * log(histRGB[i].y);
        if(histRGB[i].z > 0)
            max_valRGB.z += (histRGB[i].z / sumRGB.z) * log(histRGB[i].z);
    }

    correspRGB = max_valRGB * -1.0f;
    response[y + x * height + width * height * (disparity - 1)] = (correspRGB.x + correspRGB.y + correspRGB.z) / 3.0f;
}


__kernel void LF_CAE_Bin(
    __global float4* input,
    __global float* response,
    int UV_diameter,
    float sigma,
    int disparity,
    int depth_resolution)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int                 i,j                 ;
    int                 x_ind,y_ind         ;

    int     window_size = UV_diameter*UV_diameter               ;
    int     height_of_remap = height*UV_diameter                ;

    __local float4 center_patch  ;
    float3 histRGB[64]          ;
    float3 sumRGB = 0.0f        ;
    float3 max_valRGB = 0.0f    ;
    float3 correspRGB           ;
    float3 pixelRGB             ;
    float3 cRGB                 ;
    int3   indexRGB             ;
    float3 reducesRGB           ;
    float3 numeratorRGB         ;

    float reduces       ;
    float numerator     ;
    float denominator = 2 * pow(sigma, 2)           ;
    x_ind = ((UV_diameter+1)/2) + x * UV_diameter  ;
    y_ind = ((UV_diameter+1)/2) + y * UV_diameter  ;

    center_patch = (input[y_ind + x_ind*height_of_remap]);
    center_patch /= 4.0f    ;
    cRGB = (center_patch.xyz)     ;

    float3 weight;
    float3 floorRGB;
    float3 ceilRGB;
    float3 weightRGB;

    for (j = 0; j < UV_diameter; j++)
    {
        x_ind = j + x * UV_diameter;
        y_ind = y * UV_diameter;

        center_patch = (input[y_ind + x_ind*height_of_remap]);
        center_patch /= 4.0f    ;
        //R
        pixelRGB.x = center_patch.x             ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y             ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z             ;
        indexRGB.z = round(pixelRGB.z)  ;

        //Weight 1

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + exp(-1.0f * weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + exp(-1.0f * weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + exp(-1.0f * weight.z); //Equation (10)

        y_ind+=1;
        center_patch = (input[y_ind + x_ind*height_of_remap]);
        center_patch /= 4.0f    ;
        //R
        pixelRGB.x = center_patch.x             ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y             ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z             ;
        indexRGB.z = round(pixelRGB.z)  ;

        //Weight 1

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + exp(-1.0f * weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + exp(-1.0f * weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + exp(-1.0f * weight.z); //Equation (10)

        y_ind+=1;
        center_patch = (input[y_ind + x_ind*height_of_remap]);
        center_patch /= 4.0f    ;
        //R
        pixelRGB.x = center_patch.x             ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y             ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z             ;
        indexRGB.z = round(pixelRGB.z)  ;

        //Weight 1

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + exp(-1.0f * weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + exp(-1.0f * weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + exp(-1.0f * weight.z); //Equation (10)

        y_ind+=1;
        center_patch = (input[y_ind + x_ind*height_of_remap]);
        center_patch /= 4.0f    ;
        //R
        pixelRGB.x = center_patch.x             ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y             ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z             ;
        indexRGB.z = round(pixelRGB.z)  ;

        //Weight 1

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + exp(-1.0f * weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + exp(-1.0f * weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + exp(-1.0f * weight.z); //Equation (10)

        y_ind+=1;
        center_patch = (input[y_ind + x_ind*height_of_remap]);
        center_patch = center_patch / 4.0f    ;
        //R
        pixelRGB.x = center_patch.x             ;
        indexRGB.x = round(pixelRGB.x)  ;

        //G
        pixelRGB.y = center_patch.y             ;
        indexRGB.y = round(pixelRGB.y)  ;

        //B
        pixelRGB.z = center_patch.z             ;
        indexRGB.z = round(pixelRGB.z)  ;

        //Weight 1

        reducesRGB = (pixelRGB-cRGB)        ;
        numeratorRGB = reducesRGB * reducesRGB;
        weight = numeratorRGB / denominator ;
        weight = exp(-1.0f * weight);
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + exp(-1.0f * weight.x); //Equation (10)
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + exp(-1.0f * weight.y); //Equation (10)
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + exp(-1.0f * weight.z); //Equation (10)

        //Uniform weight
        //histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + 1; //Equation (10)
        //histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + 1; //Equation (10)
        //histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + 1; //Equation (10)


        //Weight 2
        /*
        ceilRGB = ceil(pixelRGB);
        floorRGB = floor(pixelRGB);
        reducesRGB = (pixelRGB-cRGB);
        numeratorRGB = pow(reducesRGB, 2);
        divide = numeratorRGB / denominator;

        indexRGB.x = floor(pixelRGB.x);
        indexRGB.y = floor(pixelRGB.y);
        indexRGB.z = floor(pixelRGB.z);
        weightRGB = ceilRGB - pixelRGB;
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + weightRGB.x ;
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + weightRGB.y ;
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + weightRGB.z ;

        indexRGB.x = ceil(pixelRGB.x);
        indexRGB.y = ceil(pixelRGB.y);
        indexRGB.z = ceil(pixelRGB.z);
        weightRGB = pixelRGB - floorRGB;
        histRGB[indexRGB.x].x = histRGB[indexRGB.x].x + weightRGB.x ;
        histRGB[indexRGB.y].y = histRGB[indexRGB.y].y + weightRGB.y ;
        histRGB[indexRGB.z].z = histRGB[indexRGB.z].z + weightRGB.z ;
        */
    }


    //Equation (12)
    float3 w_s = (float3)window_size;

    float4 hist_1, hist_2, hist_3;
    float4 ones = (float4)1.0f;
    #pragma unroll 16
    for (i = 0 ; i < 64 ; i+=4)
    {
        histRGB[i] = histRGB[i] / w_s;
        histRGB[i+1] = histRGB[i+1] / w_s;
        histRGB[i+2] = histRGB[i+2] / w_s;
        histRGB[i+3] = histRGB[i+3] / w_s;

        hist_1.x = histRGB[i].x;
        hist_1.y = histRGB[i+1].x;
        hist_1.z = histRGB[i+2].x;
        hist_1.w = histRGB[i+3].x;

        hist_2.x = histRGB[i].x;
        hist_2.y = histRGB[i+1].y;
        hist_2.z = histRGB[i+2].y;
        hist_2.w = histRGB[i+3].y;

        hist_3.x = histRGB[i].z;
        hist_3.y = histRGB[i+1].z;
        hist_3.z = histRGB[i+2].z;
        hist_3.w = histRGB[i+3].z;
        sumRGB.x += dot(hist_1, ones);
        sumRGB.y += dot(hist_2, ones);
        sumRGB.z += dot(hist_3, ones);
    }

    /*
    #pragma unroll
    for(i=0;i<64;i++)
    {
        histRGB[i] = histRGB[i] / w_s;
        if (histRGB[i].x != 0)
            sumRGB.x += histRGB[i].x ;
        if (histRGB[i].y != 0)
            sumRGB.y += histRGB[i].y ;
        if (histRGB[i].z != 0)
            sumRGB.z += histRGB[i].z ;
    }
    */

    #pragma unroll
    for(i=0;i<64;i++)
    {
        //|g| is histRGB[i] / sumRGB
        if(histRGB[i].x > 0)
            max_valRGB.x += log(histRGB[i].x) * (histRGB[i].x / sumRGB.x);
        if(histRGB[i].y > 0)
            max_valRGB.y += log(histRGB[i].y) * (histRGB[i].y / sumRGB.y);
        if(histRGB[i].z > 0)
            max_valRGB.z += log(histRGB[i].z) * (histRGB[i].z / sumRGB.z);
    }

    //Weight 1
    correspRGB = max_valRGB * -1.0f / 3.0f;
    //float respond = (correspRGB.x + correspRGB.y + correspRGB.z) / 3.0f;
    response[y + x * height + width * height * (disparity - 1)] = (correspRGB.x + correspRGB.y + correspRGB.z);
}

__kernel void LF_SSD(
    __global uchar4* input,
    __global float* response,
    int UV_diameter,
    float sigma,
    int disparity,
    int depth_resolution)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int                 i,j             ;
    int                 x_ind,y_ind     ;

    float   pixel                       ;
    int     index                       ;

    int     window_size = UV_diameter*UV_diameter               ;
    int     height_of_remap = height*UV_diameter                ;
    int     pixels_of_spatial = height*width  ;

    x_ind = (int)(UV_diameter/2) + x * UV_diameter  ;
    y_ind = (int)(UV_diameter/2) + y * UV_diameter  ;

    uint4 temp = convert_uint4(input[y_ind + x_ind*height_of_remap]);
    uint4 mid = temp;
    uint difference;
    uint sum = 0;

    for (i = 0; i < UV_diameter; i++)
    {
        for (j = 0; j < UV_diameter; j++)
        {
            x_ind = j + x * UV_diameter;
            y_ind = i + y * UV_diameter;
            temp = convert_uint4(input[y_ind + x_ind*height_of_remap]);
            //SAD//
            //difference = abs_diff(mid.x , temp.x);
            //SSD//
            difference = mid.x - temp.x;
            difference *= difference;
            sum += difference;
        }
    }
    response[y + x * height + width * height * (disparity - 1)] = sum;
}

/*
Weighted Median Filter (WMF)
*/
__constant float kernel_wmf_5x5[25] =
{
	1, 1, 1, 1, 1, // weighted median
	2, 2, 2, 2, 2,
	3, 3, 3, 3, 3,
	2, 2, 2, 2, 2,
	1, 1, 1, 1, 1 }; // sum = 45
__constant float kernel_wmf_3x3[9] =
{
	1, 1, 1,
	1, 4, 1,
	1, 1, 1 }; // sum = 45
// Assumes Kernel is 3x3
__kernel void LF_WMF(
	__global uchar *input,
	__global uchar *output,
	int width)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

	int iOffset, iPost, iPrev;

	iOffset = row * width;
	iPost = iOffset + width;
	iPrev = iOffset - width;

	float pixel[9];

	pixel[0] = (float)(input[iPrev + col - 1]);
	pixel[1] = (float)(input[iPrev + col]);
	pixel[2] = (float)(input[iPrev + col + 1]);

	pixel[3] = (float)(input[iOffset + col - 1]);
	pixel[4] = (float)(input[iOffset + col]);
	pixel[5] = (float)(input[iOffset + col + 1]);

	pixel[6] = (float)(input[iPost + col - 1]);
	pixel[7] = (float)(input[iPost + col]);
	pixel[8] = (float)(input[iPost + col + 1]);

	uint r0, r1, r2, r3, r4, r5, r6, r7, r8;
	r0 = pixel[0] * kernel_wmf_3x3[0];
	r1 = pixel[1] * kernel_wmf_3x3[1];
	r2 = pixel[2] * kernel_wmf_3x3[2];
	r3 = pixel[3] * kernel_wmf_3x3[3];
	r4 = pixel[4] * kernel_wmf_3x3[4];
	r5 = pixel[5] * kernel_wmf_3x3[5];
	r6 = pixel[6] * kernel_wmf_3x3[6];
	r7 = pixel[7] * kernel_wmf_3x3[7];
	r8 = pixel[8] * kernel_wmf_3x3[8];

	// perform partial bitonic sort to find current channel median
	uint uiMin = min(r0, r1);
	uint uiMax = max(r0, r1);
	r0 = uiMin;
	r1 = uiMax;

	uiMin = min(r3, r2);
	uiMax = max(r3, r2);
	r3 = uiMin;
	r2 = uiMax;

	uiMin = min(r2, r0);
	uiMax = max(r2, r0);
	r2 = uiMin;
	r0 = uiMax;

	uiMin = min(r3, r1);
	uiMax = max(r3, r1);
	r3 = uiMin;
	r1 = uiMax;

	uiMin = min(r1, r0);
	uiMax = max(r1, r0);
	r1 = uiMin;
	r0 = uiMax;

	uiMin = min(r3, r2);
	uiMax = max(r3, r2);
	r3 = uiMin;
	r2 = uiMax;

	uiMin = min(r5, r4);
	uiMax = max(r5, r4);
	r5 = uiMin;
	r4 = uiMax;

	uiMin = min(r7, r8);
	uiMax = max(r7, r8);
	r7 = uiMin;
	r8 = uiMax;

	uiMin = min(r6, r8);
	uiMax = max(r6, r8);
	r6 = uiMin;
	r8 = uiMax;

	uiMin = min(r6, r7);
	uiMax = max(r6, r7);
	r6 = uiMin;
	r7 = uiMax;

	uiMin = min(r4, r8);
	uiMax = max(r4, r8);
	r4 = uiMin;
	r8 = uiMax;

	uiMin = min(r4, r6);
	uiMax = max(r4, r6);
	r4 = uiMin;
	r6 = uiMax;

	uiMin = min(r5, r7);
	uiMax = max(r5, r7);
	r5 = uiMin;
	r7 = uiMax;

	uiMin = min(r4, r5);
	uiMax = max(r4, r5);
	r4 = uiMin;
	r5 = uiMax;

	uiMin = min(r6, r7);
	uiMax = max(r6, r7);
	r6 = uiMin;
	r7 = uiMax;

	uiMin = min(r0, r8);
	uiMax = max(r0, r8);
	r0 = uiMin;
	r8 = uiMax;

	r4 = max(r0, r4);
	r5 = max(r1, r5);

	r6 = max(r2, r6);
	r7 = max(r3, r7);

	r4 = min(r4, r6);
	r5 = min(r5, r7);
	uchar uiResult = 0;

	uiResult |= (uchar)min(r4, r5);
	output[iOffset + col] = (uiResult);
}

/*
Winner Takes All (WTA)
*/
__kernel void LF_WTA_Lytro(
    __global float *datacost,
    __global uchar *dispFinal,
    int dispRange,
    float dispScale)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	float minCost = FLT_MAX;
	float disparity = 0;
	for (int d = 0; d < dispRange; d++)
	{
		float cost = datacost[y + x * height + width * height * (d)];
		if (minCost > cost)
		{
			minCost = cost;
			disparity = (float)d;
		}
	}
	dispFinal[y + x * height] =  255 - (uchar)(disparity * dispScale);
}

__kernel void LF_WTA_Image(
    __global float *datacost,
    __global uchar *dispFinal,
    int dispRange,
    float dispScale)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	float minCost = FLT_MAX;
	float disparity = 0;
	for (int d = 0; d < dispRange; d++)
	{
		float cost = datacost[y + x * height + width * height * (d)];
		if (minCost > cost)
		{
			minCost = cost;
			disparity = (float)d;
		}
	}
	dispFinal[y + x * height] =  (uchar)(disparity * dispScale);
}