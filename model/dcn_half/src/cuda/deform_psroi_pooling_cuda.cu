/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>

// ---------------------------------------------------------------------------
// _to_float — convert any scalar_t to float without C++ conversion operators
// (blocked by -D__CUDA_NO_HALF* on Windows)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float _psroi_to_float(float x) { return x; }
__device__ __forceinline__ float _psroi_to_float(double x) { return (float)x; }
__device__ __forceinline__ float _psroi_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float _psroi_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float _psroi_to_float(c10::Half x) {
  __half raw;
  memcpy(&raw, &x, sizeof(__half));
  return __half2float(raw);
}
__device__ __forceinline__ float _psroi_to_float(c10::BFloat16 x) {
  __nv_bfloat16 raw;
  memcpy(&raw, &x, sizeof(__nv_bfloat16));
  return __bfloat162float(raw);
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Bilinear interpolation — works with any scalar_t via float internally
template <typename T>
__device__ float _psroi_bilinear_interp(
    const T *data,
    float x, float y,
    const int width,
    const int height)
{
  int x1 = (int)floorf(x);
  int x2 = (int)ceilf(x);
  int y1 = (int)floorf(y);
  int y2 = (int)ceilf(y);
  float dist_x = x - x1;
  float dist_y = y - y1;
  float value11 = _psroi_to_float(data[y1 * width + x1]);
  float value12 = _psroi_to_float(data[y2 * width + x1]);
  float value21 = _psroi_to_float(data[y1 * width + x2]);
  float value22 = _psroi_to_float(data[y2 * width + x2]);
  float value = (1.0f - dist_x) * (1.0f - dist_y) * value11 +
                (1.0f - dist_x) * dist_y * value12 +
                dist_x * (1.0f - dist_y) * value21 +
                dist_x * dist_y * value22;
  return value;
}

template <typename T>
__global__ void DeformablePSROIPoolForwardKernel(
    const int count,
    const T *bottom_data,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const T *bottom_rois, const T *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    T *top_data,
    T *top_count)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
    float roi_start_w = roundf(_psroi_to_float(offset_bottom_rois[1])) * spatial_scale - 0.5f;
    float roi_start_h = roundf(_psroi_to_float(offset_bottom_rois[2])) * spatial_scale - 0.5f;
    float roi_end_w = (roundf(_psroi_to_float(offset_bottom_rois[3])) + 1.0f) * spatial_scale - 0.5f;
    float roi_end_h = (roundf(_psroi_to_float(offset_bottom_rois[4])) + 1.0f) * spatial_scale - 0.5f;

    // Force too small ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 0.1f);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1f);

    // Compute w and h at bottom
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);

    int part_h = (int)floorf(static_cast<float>(ph) / pooled_height * part_size);
    int part_w = (int)floorf(static_cast<float>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;

    float trans_x = no_trans ? 0.0f : _psroi_to_float(bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w]) * trans_std;
    float trans_y = no_trans ? 0.0f : _psroi_to_float(bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w]) * trans_std;

    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    float sum = 0;
    int count = 0;
    int gw = (int)floorf(static_cast<float>(pw) * group_size / pooled_width);
    int gh = (int)floorf(static_cast<float>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    const T *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        float w = wstart + iw * sub_bin_size_w;
        float h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5f || w > width - 0.5f || h < -0.5f || h > height - 0.5f)
        {
          continue;
        }
        w = fminf(fmaxf(w, 0.0f), width - 1.0f);
        h = fminf(fmaxf(h, 0.0f), height - 1.0f);
        int c = (ctop * group_size + gh) * group_size + gw;
        float val = _psroi_bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        count++;
      }
    }
    top_data[index] = count == 0 ? static_cast<T>(0) : static_cast<T>(sum / count);
    top_count[index] = static_cast<T>(count);
  }
}

template <typename T>
__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count,
    const T *top_diff,
    const T *top_count,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    T *bottom_data_diff, T *bottom_trans_diff,
    const T *bottom_data,
    const T *bottom_rois,
    const T *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
    float roi_start_w = roundf(_psroi_to_float(offset_bottom_rois[1])) * spatial_scale - 0.5f;
    float roi_start_h = roundf(_psroi_to_float(offset_bottom_rois[2])) * spatial_scale - 0.5f;
    float roi_end_w = (roundf(_psroi_to_float(offset_bottom_rois[3])) + 1.0f) * spatial_scale - 0.5f;
    float roi_end_h = (roundf(_psroi_to_float(offset_bottom_rois[4])) + 1.0f) * spatial_scale - 0.5f;

    // Force too small ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 0.1f);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1f);

    // Compute w and h at bottom
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);

    int part_h = (int)floorf(static_cast<float>(ph) / pooled_height * part_size);
    int part_w = (int)floorf(static_cast<float>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;

    float trans_x = no_trans ? 0.0f : _psroi_to_float(bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w]) * trans_std;
    float trans_y = no_trans ? 0.0f : _psroi_to_float(bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w]) * trans_std;

    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (static_cast<float>(top_count[index]) <= 0.0f)
    {
      continue;
    }
    float diff_val = _psroi_to_float(top_diff[index]) / _psroi_to_float(top_count[index]);
    const T *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
    T *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = (int)floorf(static_cast<float>(pw) * group_size / pooled_width);
    int gh = (int)floorf(static_cast<float>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        float w = wstart + iw * sub_bin_size_w;
        float h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5f || w > width - 0.5f || h < -0.5f || h > height - 0.5f)
        {
          continue;
        }
        w = fminf(fmaxf(w, 0.0f), width - 1.0f);
        h = fminf(fmaxf(h, 0.0f), height - 1.0f);
        int c = (ctop * group_size + gh) * group_size + gw;
        // backward on feature
        int x0 = (int)floorf(w);
        int x1 = (int)ceilf(w);
        int y0 = (int)floorf(h);
        int y1 = (int)ceilf(h);
        float dist_x = w - x0, dist_y = h - y0;
        float q00 = (1.0f - dist_x) * (1.0f - dist_y);
        float q01 = (1.0f - dist_x) * dist_y;
        float q10 = dist_x * (1.0f - dist_y);
        float q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, static_cast<T>(q00 * diff_val));
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, static_cast<T>(q01 * diff_val));
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, static_cast<T>(q10 * diff_val));
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, static_cast<T>(q11 * diff_val));

        if (no_trans)
        {
          continue;
        }
        float U00 = _psroi_to_float(offset_bottom_data[bottom_index_base + y0 * width + x0]);
        float U01 = _psroi_to_float(offset_bottom_data[bottom_index_base + y1 * width + x0]);
        float U10 = _psroi_to_float(offset_bottom_data[bottom_index_base + y0 * width + x1]);
        float U11 = _psroi_to_float(offset_bottom_data[bottom_index_base + y1 * width + x1]);
        float diff_x = (U11 * dist_y + U10 * (1.0f - dist_y) - U01 * dist_y - U00 * (1.0f - dist_y)) * trans_std * diff_val;
        diff_x *= roi_width;
        float diff_y = (U11 * dist_x + U01 * (1.0f - dist_x) - U10 * dist_x - U00 * (1.0f - dist_x)) * trans_std * diff_val;
        diff_y *= roi_height;

        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, static_cast<T>(diff_x));
        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, static_cast<T>(diff_y));
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_cuda_forward(const at::Tensor &input,
                                  const at::Tensor &bbox,
                                  const at::Tensor &trans,
                                  const int no_trans,
                                  const float spatial_scale,
                                  const int output_dim,
                                  const int group_size,
                                  const int pooled_size,
                                  const int part_size,
                                  const int sample_per_part,
                                  const float trans_std)
{
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(bbox.is_cuda(), "rois must be a CUDA tensor");
  AT_ASSERTM(trans.is_cuda(), "trans must be a CUDA tensor");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);
  const int num_bbox = bbox.size(0);

  AT_ASSERTM(channels == output_dim, "input channels and output channels must equal");
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;

  auto out = at::empty({num_bbox, output_dim, pooled_height, pooled_width}, input.options());
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  auto top_count = at::zeros({num_bbox, output_dim, pooled_height, pooled_width}, input.options());

  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (out.numel() == 0)
  {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(out, top_count);
  }

  dim3 grid(std::min(static_cast<long>(ceil_div(out_size, 512L)), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "deform_psroi_pooling_cuda_forward", [&] {
    DeformablePSROIPoolForwardKernel<scalar_t><<<grid, block, 0, stream>>>(
        out_size,
        input.contiguous().data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height, width,
        pooled_height,
        pooled_width,
        bbox.contiguous().data_ptr<scalar_t>(),
        trans.contiguous().data_ptr<scalar_t>(),
        no_trans,
        trans_std,
        sample_per_part,
        output_dim,
        group_size,
        part_size,
        num_classes,
        channels_each_class,
        out.data_ptr<scalar_t>(),
        top_count.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(out, top_count);
}

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_cuda_backward(const at::Tensor &out_grad,
                                   const at::Tensor &input,
                                   const at::Tensor &bbox,
                                   const at::Tensor &trans,
                                   const at::Tensor &top_count,
                                   const int no_trans,
                                   const float spatial_scale,
                                   const int output_dim,
                                   const int group_size,
                                   const int pooled_size,
                                   const int part_size,
                                   const int sample_per_part,
                                   const float trans_std)
{
  AT_ASSERTM(out_grad.is_cuda(), "out_grad must be a CUDA tensor");
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(bbox.is_cuda(), "bbox must be a CUDA tensor");
  AT_ASSERTM(trans.is_cuda(), "trans must be a CUDA tensor");
  AT_ASSERTM(top_count.is_cuda(), "top_count must be a CUDA tensor");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);
  const int num_bbox = bbox.size(0);

  AT_ASSERTM(channels == output_dim, "input channels and output channels must equal");
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  auto input_grad = at::zeros({batch, channels, height, width}, out_grad.options());
  auto trans_grad = at::zeros_like(trans);

  if (input_grad.numel() == 0)
  {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(input_grad, trans_grad);
  }

  dim3 grid(std::min(static_cast<long>(ceil_div(out_size, 512L)), 4096L));
  dim3 block(512);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, out_grad.scalar_type(), "deform_psroi_pooling_cuda_backward", [&] {
    DeformablePSROIPoolBackwardAccKernel<scalar_t><<<grid, block, 0, stream>>>(
        out_size,
        out_grad.contiguous().data_ptr<scalar_t>(),
        top_count.contiguous().data_ptr<scalar_t>(),
        num_bbox,
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        output_dim,
        input_grad.contiguous().data_ptr<scalar_t>(),
        trans_grad.contiguous().data_ptr<scalar_t>(),
        input.contiguous().data_ptr<scalar_t>(),
        bbox.contiguous().data_ptr<scalar_t>(),
        trans.contiguous().data_ptr<scalar_t>(),
        no_trans,
        trans_std,
        sample_per_part,
        group_size,
        part_size,
        num_classes,
        channels_each_class);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(input_grad, trans_grad);
}
