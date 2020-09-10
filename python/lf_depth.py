import os
import time
import numpy as np
import pyopencl as cl

# image io & visualization tool
import cv2
from matplotlib import pyplot as plt

from tqdm import tqdm

class LF_Depth():
    def __init__(self):
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        self.ctx = cl.Context([device])      # Create a context with your device
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        CL_SOURCE = open(os.path.join(os.getcwd(), 'lf_depth.cl'), 'r').read()

        self.prg = cl.Program(self.ctx, CL_SOURCE).build(options='-cl-mad-enable -cl-fast-relaxed-math')

    def process(self,
        input_img,
        depth_resolution=75,
        delta=0.02,
        UV_diameter=5,
        UV_radius=2,
        sigma=10.0,
        thread_num=(16, 16),
        is_image=True,
        method='CAE',
        post_proc=True
        ):
        # delta: 0.0214 MONA || 0.0316 PAP || 0.0324 B2 || 0.0416 B1 || 0.0732 LIFE || 0.0518 Medieval
        assert len(input_img.shape) == 3, "Unavailable input image shape:{}, only [Ch, H, W] is available".format(input_img.shape)

        # size check
        h, w, ch = input_img.shape
        if ch == 3:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGBA)
        assert w % thread_num[0] == 0 and h % thread_num[1] == 0, "Invalid width & height"

        # spatial range
        w_spatial, h_spatial = (w // UV_diameter, h // UV_diameter)

        # kernel arguments
        var_w_spatial = np.int32(w_spatial)
        var_h_spatial = np.int32(h_spatial)
        var_delta = np.float32(delta)
        var_UV_diameter = np.int32(UV_diameter)
        var_UV_radius = np.int32(UV_radius)
        var_sigma = np.float32(sigma)
        var_depth_resolution = np.int32(depth_resolution)

        # cl.Image2D settings
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        # device memory object
        imageLF = cl.Image(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(w, h), hostbuf=input_img)
        bufferRemap = cl.Buffer(self.ctx, mf.READ_WRITE, size=w*h*4*4) # size -> w * h * clfoat4(= 4 * 4 bytes)
        bufferResponse = cl.Buffer(self.ctx, mf.READ_WRITE, size=w_spatial*h_spatial*depth_resolution*4) # size -> w * h * depth_resolution * clfoat(= 4 bytes)
        bufferDepth = cl.Buffer(self.ctx, mf.READ_WRITE, size=w_spatial*h_spatial) # size -> w * h * cluchar(= 1 bytes)
        bufferDepth_Refined = cl.Buffer(self.ctx, mf.READ_WRITE, size=w_spatial*h_spatial) # size -> w * h * cluchar(= 1 bytes)
        
        # final output
        output_img = np.zeros((h_spatial, w_spatial)).astype(np.uint8)

        # select input type & algorithm
        if is_image:
            gRemap = self.prg.LF_Remap_Image
            gWTA = self.prg.LF_WTA_Image
        else:
            gRemap = self.prg.LF_Remap_Lytro
            gWTA = self.prg.LF_WTA_Lytro

        if method == 'CAE':
            gCost = self.prg.LF_CAE
        elif method == 'CAE_Bin':
            gCost = self.prg.LF_CAE_Bin
        else:
            gCost = self.prg.LF_SSD

        for idx in tqdm(range(1, depth_resolution + 1)):
            alpha = -((idx - (depth_resolution + 1) / 2) * (delta))
            #print("alpha", alpha)
            var_alpha = np.float32(alpha)
            gRemap(
                self.queue,
                (w_spatial, h_spatial),
                (thread_num[0], thread_num[1]),
                imageLF,
                bufferRemap,
                var_delta,
                var_UV_diameter,
                var_UV_radius,
                var_alpha
            ).wait()

            # kernel argument for WTA : disparity index
            var_index = np.int32(idx)
            gCost(
                self.queue,
                (w_spatial, h_spatial),
                (thread_num[0], thread_num[1]),
                bufferRemap,
                bufferResponse,
                var_UV_diameter,
                var_sigma,
                var_index,
                var_depth_resolution
            ).wait()

        # kernel argument for WTA : scale factor
        var_scale = np.float32(256.0 / depth_resolution)
        gWTA(
            self.queue,
            (w_spatial, h_spatial),
            (thread_num[0], thread_num[1]),
            bufferResponse,
            bufferDepth,
            var_depth_resolution,
            var_scale
        ).wait()

        if post_proc:
            self.prg.LF_WMF(
                self.queue,
                (w_spatial - 2, h_spatial - 2), # To avoid boundary-problem
                None, # local size -> cl::NullRange
                bufferDepth,
                bufferDepth_Refined,
                var_w_spatial,
                global_offset=[1, 1] # offset
            ).wait()
            cl.enqueue_copy(self.queue, output_img, bufferDepth_Refined, is_blocking=True)
        else:
            cl.enqueue_copy(self.queue, output_img, bufferDepth, is_blocking=True)
        return output_img    

def demo(input_img_path):
    lfdepth = LF_Depth()
    input_img = cv2.imread(input_img_path)
    # method should be one of {"CAE", "CAE_Bin", "SSD"}
    output_img = lfdepth.process(input_img, is_image=True, method='CAE', post_proc=True)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()
    axs[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGBA))
    axs[1].imshow(output_img, cmap='gray')
    plt.show()

if __name__ == '__main__' :
    demo("../LightField/Mona_Crop.png")

