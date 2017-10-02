#if GOOGLE_CUDA

#include <iostream>
#include <stdio.h>
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// CUDA: index helpers
#define idx4_5(index, d1, d2, d3, d4, d5) (index % d5)
#define idx4_4(index, d1, d2, d3, d4, d5) ((index / d5) % d4)
#define idx4_3(index, d1, d2, d3, d4, d5) ((index / d5 / d4) % d3)
#define idx4_2(index, d1, d2, d3, d4, d5) ((index / d5 / d4 / d3) % d2)
#define idx4_1(index, d1, d2, d3, d4, d5) ((index / d5 / d4 / d3 / d2) %d1)

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      return 1; \
    } \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  // TODO rewrite this part to be consistent with tf conventions
  int optimal_number_of_blocks = (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  int max_number_of_blocks = 65000;
  return std::min(optimal_number_of_blocks, max_number_of_blocks);
}


#define Dtype float

__global__ void RoiPooling3DKernel(const Dtype* input, const int* rois,
                                 int n_rois, int channels, int height, int width, int depth,
                                 int pooled_height, int pooled_width, int pooled_depth,
                                 Dtype* output, int* argmax_output) {
    int output_size = n_rois * channels * pooled_height * pooled_width * pooled_depth;

    CUDA_KERNEL_LOOP(index, output_size) {
    // (n, c, ph, pw, pd) is an element in the pooled output
    int pd = idx4_5(index, n_rois, channels, pooled_height, pooled_width, pooled_depth);
    int pw = idx4_4(index, n_rois, channels, pooled_height, pooled_width, pooled_depth);
    int ph = idx4_3(index, n_rois, channels, pooled_height, pooled_width, pooled_depth);
    int c  = idx4_2(index, n_rois, channels, pooled_height, pooled_width, pooled_depth);
    int n  = idx4_1(index, n_rois, channels, pooled_height, pooled_width, pooled_depth);

    auto bottom_rois_act = rois + n * 5;

    int roi_batch_ind = bottom_rois_act[0];
    int roi_start_d = bottom_rois_act[1];
    int roi_start_w = bottom_rois_act[2];
    int roi_start_h = bottom_rois_act[3];
    int roi_end_d = bottom_rois_act[4];
    int roi_end_w = bottom_rois_act[5];
    int roi_end_h = bottom_rois_act[6];

    // Force malformed ROIs to be 1x1
    // NOTE(maciek): roi_start, roi_end seems to be inclusive
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_depth  = max(roi_end_d - roi_start_d + 1, 1);

    // divide the ROIs into smaller regions for max pooling
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);
    Dtype bin_size_d = static_cast<Dtype>(roi_depth) / static_cast<Dtype>(pooled_depth);

    // compute the precise coordinates of each pooling subregion of the ROIs
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
    int dstart = static_cast<int>(floor(static_cast<Dtype>(pd) * bin_size_d));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));
    int dend = static_cast<int>(ceil(static_cast<Dtype>(pd + 1) * bin_size_d));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    dstart = min(max(dstart + roi_start_d, 0), depth);
    dend = min(max(dend + roi_start_d, 0), depth);

    //printf("%d %d %d %d %d %d %d %d\n", n, c, pw, ph, hstart, hend, wstart, wend);

    bool is_empty = (hend <= hstart) || (wend <= wstart) || (dend <= dstart);

    // Define an empty pooling region to be zero

    Dtype maxval = is_empty ? 0 : -999999999.0;
    //Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd

    int maxidx = -1;
    auto input_act = input + (roi_batch_ind * height * width * depth * channels);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	for (int d = dstart; d < dend; ++d) {
	  int bottom_index = (h * width * depth + w * depth + d) * channels + c;

	  // bottom index is relative to 2d image only
	  if (input_act[bottom_index] > maxval) {
	    maxval = input_act[bottom_index];
	    maxidx = bottom_index;
	  }
	}
      }
    }
    output[index] = maxval;
    argmax_output[index] = maxidx;
  }
}


void RoiPooling3DKernelLauncher(const float* input, const int* rois, int n_rois, int channels, int height, int width, int depth,
				int pooled_height, int pooled_width, int pooled_depth, Dtype* output, int* argmax_output) {
    int out_size = n_rois * channels * pooled_height * pooled_width * pooled_depth;

    RoiPooling3DKernel<<<CAFFE_GET_BLOCKS(out_size), CAFFE_CUDA_NUM_THREADS>>>(input, rois, n_rois, channels, height, width, depth,
									       pooled_height, pooled_width, pooled_depth, output, argmax_output);
}


/////////////// Grad
__global__ void RoiPooling3DGradKernel(const Dtype* orig_input, const int* orig_rois,
                                 int mb_size,
				 int n_rois, int channels, int height, int width, int depth,
				 int pooled_height, int pooled_width, int pooled_depth,
                                 const Dtype* orig_output, const int* orig_argmax_output,
                                 const Dtype* orig_output_grad,
                                 Dtype* output) {

    int orig_input_size = mb_size * height * width * depth * channels;

    CUDA_KERNEL_LOOP(index, orig_input_size) {
    // (n, h, w, d, c) coords in bottom data
    int c = idx4_5(index, mb_size, height, width, depth, channels);
    int d = idx4_4(index, mb_size, height, width, depth, channels);
    int w = idx4_3(index, mb_size, height, width, depth, channels);
    int h = idx4_2(index, mb_size, height, width, depth, channels);
    int n = idx4_1(index, mb_size, height, width, depth, channels);

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < n_rois; ++roi_n) {
      const int* offset_bottom_rois = orig_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_d = offset_bottom_rois[1];
      int roi_start_w = offset_bottom_rois[2];
      int roi_start_h = offset_bottom_rois[3];
      int roi_end_d = offset_bottom_rois[4];
      int roi_end_w = offset_bottom_rois[5];
      int roi_end_h = offset_bottom_rois[6];

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h &&
			   d >= roi_start_d && d <= roi_end_d);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width * pooled_depth;
      const Dtype* offset_top_diff = orig_output_grad + offset;
      const int* offset_argmax_data = orig_argmax_output + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_depth = max(roi_end_d - roi_start_d + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);
      Dtype bin_size_d = static_cast<Dtype>(roi_depth) / static_cast<Dtype>(pooled_depth);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);
      int pdstart = floor(static_cast<Dtype>(d - roi_start_d) / bin_size_d);
      int pdend = ceil(static_cast<Dtype>(d - roi_start_d + 1) / bin_size_d);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);
      pdstart = min(max(pdstart, 0), pooled_depth);
      pdend = min(max(pdend, 0), pooled_depth);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
	  for (int pd = pdstart; pd < pdend; ++pd) {
	    if (offset_argmax_data[ph * pooled_width * pooled_depth + pw * pooled_depth + pd] == (h * width * depth + w * depth + d)) {
	      gradient += offset_top_diff[ph * pooled_width * pooled_depth + pw * pooled_depth + pd];
	    }
	  }
        }
      }
    }
    output[index] = gradient;
  }

}

void RoiPooling3DGradKernelLauncher(const Dtype* orig_input, const int* orig_rois,
                                 int mb_size,
				 int n_rois, int channels, int height, int width, int depth,
				 int pooled_height, int pooled_width, int pooled_depth,
                                 const Dtype* orig_output, const int* orig_argmax_output,
                                 const Dtype* orig_output_grad,
                                 Dtype* output) {
    int out_size = mb_size * height * width * depth * channels;
    RoiPooling3DGradKernel<<<CAFFE_GET_BLOCKS(out_size), CAFFE_CUDA_NUM_THREADS>>>(orig_input, orig_rois,
										 mb_size, n_rois, channels, height, width, depth, pooled_height, pooled_width, pooled_depth,
        orig_output, orig_argmax_output, orig_output_grad, output);
}

#endif
