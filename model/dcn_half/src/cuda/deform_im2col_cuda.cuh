#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// Forward declarations for device functions used in kernels below.
// CUDA 13+ nvcc requires device functions to be declared before use,
// even inside templated global kernels (lazy instantiation is not enough).
template <typename scalar_t>
__device__ void dmcn_add_bilinear(scalar_t *grad_im, const int height, const int width,
                                  scalar_t h, scalar_t w, const int h_end, const int w_end,
                                  scalar_t val);

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// ---------------------------------------------------------------------------
// NOTE: All floating-point arithmetic inside kernels uses float internally.
// Static casts on __half/__nv_bfloat16 are done via overloaded _to_float()
// to avoid depending on C++ conversion operators (blocked by -D__CUDA_NO_HALF*).
// ---------------------------------------------------------------------------
// Convert scalar_t to float for arithmetic without using C++ conversion
// operators (blocked by -D__CUDA_NO_HALF* on Windows).
// Uses C API __half2float/__bfloat162float for raw __half/__nv_bfloat16;
// for c10::Half/c10::BFloat16 we bitcast to the underlying type first.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float _to_float(T x);

template<> __device__ __forceinline__ float _to_float(float x) { return x; }
template<> __device__ __forceinline__ float _to_float(double x) { return (float)x; }
template<> __device__ __forceinline__ float _to_float(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float _to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
// c10::Half/BFloat16 have the same representation as __half/__nv_bfloat16
// (16-bit unsigned short). Use memcpy to get raw bits and convert.
template<> __device__ __forceinline__ float _to_float(c10::Half x) {
  unsigned short us;
  memcpy(&us, &x, sizeof(unsigned short));
  __half h;
  memcpy(&h, &us, sizeof(__half));
  return __half2float(h);
}
template<> __device__ __forceinline__ float _to_float(c10::BFloat16 x) {
  unsigned short us;
  memcpy(&us, &x, sizeof(unsigned short));
  __nv_bfloat16 h;
  memcpy(&h, &us, sizeof(__nv_bfloat16));
  return __bfloat162float(h);
}

__device__ __forceinline__ float _scalar_floor(float x) { return floorf(x); }
__device__ __forceinline__ float _scalar_abs(float x)  { return fabsf(x); }
__device__ __forceinline__ double _scalar_floor(double x) { return floor(x); }
__device__ __forceinline__ double _scalar_abs(double x)  { return fabs(x); }
__device__ __forceinline__ float _scalar_floor(__half x) { return floorf(__half2float(x)); }
__device__ __forceinline__ float _scalar_abs(__half x)  { return fabsf(__half2float(x)); }
__device__ __forceinline__ float _scalar_floor(__nv_bfloat16 x) { return floorf(__bfloat162float(x)); }
__device__ __forceinline__ float _scalar_abs(__nv_bfloat16 x)  { return fabsf(__bfloat162float(x)); }

// ---------------------------------------------------------------------------
// Bilinear interpolation — works with float/double/half/bf16
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ float _bilinear_fetch(const scalar_t *bottom_data, const int data_width,
                                 int h, int w, int height, int width) {
  if (h >= 0 && h < height && w >= 0 && w < width)
    return _to_float(bottom_data[h * data_width + w]);
  return 0.0f;
}

template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                      const int height, const int width, scalar_t h, scalar_t w)
{
  float h_f = _to_float(h);
  float w_f = _to_float(w);
  int h_low = (int)floorf(h_f);
  int w_low = (int)floorf(w_f);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h_f - h_low;
  float lw = w_f - w_low;
  float hhf = 1.0f - lh, hwf = 1.0f - lw;

  float v1 = _bilinear_fetch(bottom_data, data_width, h_low, w_low, height, width);
  float v2 = _bilinear_fetch(bottom_data, data_width, h_low, w_high, height, width);
  float v3 = _bilinear_fetch(bottom_data, data_width, h_high, w_low, height, width);
  float v4 = _bilinear_fetch(bottom_data, data_width, h_high, w_high, height, width);

  float val = (hhf * hwf * v1 + hhf * lw * v2 + lh * hwf * v3 + lh * lw * v4);
  return static_cast<scalar_t>(val);
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  float ah = _to_float(argmax_h);
  float aw = _to_float(argmax_w);
  if (ah <= -1.0f || ah >= static_cast<float>(height) || aw <= -1.0f || aw >= static_cast<float>(width))
    return static_cast<scalar_t>(0);

  int argmax_h_low = (int)floorf(ah);
  int argmax_w_low = (int)floorf(aw);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0.0f;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1.0f - ah) * (w + 1.0f - aw);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1.0f - ah) * (aw + 1.0f - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (ah + 1.0f - h) * (w + 1.0f - aw);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (ah + 1.0f - h) * (aw + 1.0f - w);
  return static_cast<scalar_t>(weight);
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                            const int height, const int width, const scalar_t *im_data,
                                            const int data_width, const int bp_dir)
{
  float ah = _to_float(argmax_h);
  float aw = _to_float(argmax_w);
  if (ah <= -1.0f || ah >= static_cast<float>(height) || aw <= -1.0f || aw >= static_cast<float>(width))
    return static_cast<scalar_t>(0);

  int argmax_h_low = (int)floorf(ah);
  int argmax_w_low = (int)floorf(aw);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0.0f;

  if (bp_dir == 0)
  {
    float v11 = _bilinear_fetch(im_data, data_width, argmax_h_low, argmax_w_low, height, width);
    float v12 = _bilinear_fetch(im_data, data_width, argmax_h_low, argmax_w_high, height, width);
    float v21 = _bilinear_fetch(im_data, data_width, argmax_h_high, argmax_w_low, height, width);
    float v22 = _bilinear_fetch(im_data, data_width, argmax_h_high, argmax_w_high, height, width);

    weight += -1.0f * (argmax_w_low + 1.0f - aw) * v11;
    weight += -1.0f * (aw - argmax_w_low) * v12;
    weight += (argmax_w_low + 1.0f - aw) * v21;
    weight += (aw - argmax_w_low) * v22;
  }
  else if (bp_dir == 1)
  {
    float v11 = _bilinear_fetch(im_data, data_width, argmax_h_low, argmax_w_low, height, width);
    float v12 = _bilinear_fetch(im_data, data_width, argmax_h_low, argmax_w_high, height, width);
    float v21 = _bilinear_fetch(im_data, data_width, argmax_h_high, argmax_w_low, height, width);
    float v22 = _bilinear_fetch(im_data, data_width, argmax_h_high, argmax_w_high, height, width);

    weight += -1.0f * (argmax_h_low + 1.0f - ah) * v11;
    weight += (argmax_h_low + 1.0f - ah) * v12;
    weight += -1.0f * (ah - argmax_h_low) * v21;
    weight += (ah - argmax_h_low) * v22;
  }

  return static_cast<scalar_t>(weight);
}

// ---------------------------------------------------------------------------
// Half atomics via CAS — forward declarations (no arch guard: needed for
// host compilation phase of `if constexpr` branches in dmcn_add_bilinear).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void _dcn_atomic_add_half(T *addr, T val);
template <typename T>
__device__ __forceinline__ void _dcn_atomic_add_bf16(T *addr, T val);

// ---------------------------------------------------------------------------
// Half atomics via CAS — definitions (guarded by arch for SM requirements)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template <typename T>
__device__ __forceinline__ void _dcn_atomic_add_half(T *addr, T val) {
  unsigned int *addr_as_ui = (unsigned int *)((char *)addr - ((size_t)addr & 2));
  unsigned int old = *addr_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short old_h = (old >> (((size_t)addr & 2) * 8U)) & 0xffff;
    __half h_old;
    memcpy(&h_old, &old_h, sizeof(__half));
    __half h_val;
    memcpy(&h_val, &val, sizeof(__half));
    __half h_new = __hadd(h_old, h_val);
    unsigned short new_h;
    memcpy(&new_h, &h_new, sizeof(unsigned short));
    old = (old & ~(0xffffU << (((size_t)addr & 2) * 8U))) |
          ((unsigned int)new_h << (((size_t)addr & 2) * 8U));
    old = atomicCAS(addr_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <typename T>
__device__ __forceinline__ void _dcn_atomic_add_bf16(T *addr, T val) {
  unsigned int *addr_as_ui = (unsigned int *)((char *)addr - ((size_t)addr & 2));
  unsigned int old = *addr_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short old_h = (old >> (((size_t)addr & 2) * 8U)) & 0xffff;
    __nv_bfloat16 h_old;
    memcpy(&h_old, &old_h, sizeof(__nv_bfloat16));
    __nv_bfloat16 h_val;
    memcpy(&h_val, &val, sizeof(__nv_bfloat16));
    __nv_bfloat16 h_new = __hadd(h_old, h_val);
    unsigned short new_h;
    memcpy(&new_h, &h_new, sizeof(unsigned short));
    old = (old & ~(0xffffU << (((size_t)addr & 2) * 8U))) |
          ((unsigned int)new_h << (((size_t)addr & 2) * 8U));
    old = atomicCAS(addr_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n,
                                             const scalar_t *data_im, const scalar_t *data_offset,
                                             const int height, const int width, const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w,
                                             const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w,
                                             const int channel_per_deformable_group,
                                             const int batch_size, const int num_channels, const int deformable_group,
                                             const int height_col, const int width_col,
                                             scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        float h_im_f = _to_float(h_im);
        float w_im_f = _to_float(w_im);
        if (h_im_f > -1.0f && w_im_f > -1.0f && h_im_f < static_cast<float>(height) && w_im_f < static_cast<float>(width))
        {
          // if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Backward kernel: col2im (gradient w.r.t. input)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(const int n,
                                             const scalar_t *data_col, const scalar_t *data_offset,
                                             const int channels, const int height, const int width,
                                             const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w,
                                             const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w,
                                             const int channel_per_deformable_group,
                                             const int batch_size, const int deformable_group,
                                             const int height_col, const int width_col,
                                             scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *grad_im_ptr = grad_im + (b_col * channels + c_im) * height * width;
    const scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        float h_im_f = _to_float(h_im);
        float w_im_f = _to_float(w_im);

        if (h_im_f > -1.0f && w_im_f > -1.0f && h_im_f < static_cast<float>(height) && w_im_f < static_cast<float>(width))
        {
          dmcn_add_bilinear(grad_im_ptr, height, width, h_im, w_im, height, width, data_col_ptr[0]);
        }
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename scalar_t>
__device__ void dmcn_add_bilinear(scalar_t *grad_im, const int height, const int width,
                                  scalar_t h, scalar_t w, const int h_end, const int w_end,
                                  scalar_t val) {
  float h_f = _to_float(h);
  float w_f = _to_float(w);
  int h_low = (int)floorf(h_f);
  int w_low = (int)floorf(w_f);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h_f - h_low;
  float lw = w_f - w_low;
  float hhf = 1.0f - lh;
  float hwf = 1.0f - lw;
  float v = _to_float(val);

  if constexpr (sizeof(scalar_t) == 2) {
    _dcn_atomic_add_half<scalar_t>(grad_im + h_low * width + w_low, static_cast<scalar_t>(hhf * hwf * v));
    _dcn_atomic_add_half<scalar_t>(grad_im + h_low * width + w_high, static_cast<scalar_t>(hhf * lw * v));
    _dcn_atomic_add_half<scalar_t>(grad_im + h_high * width + w_low, static_cast<scalar_t>(lh * hwf * v));
    _dcn_atomic_add_half<scalar_t>(grad_im + h_high * width + w_high, static_cast<scalar_t>(lh * lw * v));
  } else {
    atomicAdd(grad_im + h_low * width + w_low, static_cast<scalar_t>(hhf * hwf * v));
    atomicAdd(grad_im + h_low * width + w_high, static_cast<scalar_t>(hhf * lw * v));
    atomicAdd(grad_im + h_high * width + w_low, static_cast<scalar_t>(lh * hwf * v));
    atomicAdd(grad_im + h_high * width + w_high, static_cast<scalar_t>(lh * lw * v));
  }
}

// ---------------------------------------------------------------------------
// Backward kernel: col2im_coord (gradient w.r.t. offset)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n,
                                                   const scalar_t *data_col, const scalar_t *data_im,
                                                   const scalar_t *data_offset,
                                                   const int channels, const int height, const int width,
                                                   const int kernel_h, const int kernel_w,
                                                   const int pad_h, const int pad_w,
                                                   const int stride_h, const int stride_w,
                                                   const int dilation_h, const int dilation_w,
                                                   const int channel_per_deformable_group,
                                                   const int batch_size, const int offset_channels, const int deformable_group,
                                                   const int height_col, const int width_col,
                                                   scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      float inv_h = static_cast<float>(h_in + i * dilation_h) + _to_float(offset_h);
      float inv_w = static_cast<float>(w_in + j * dilation_w) + _to_float(offset_w);
      if (inv_h <= -1.0f || inv_w <= -1.0f || inv_h >= static_cast<float>(height) || inv_w >= static_cast<float>(width))
      {
        inv_h = inv_w = -2.0f;
      }
      const scalar_t weight = dmcn_get_coordinate_weight(
          static_cast<scalar_t>(inv_h), static_cast<scalar_t>(inv_w),
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }
    grad_offset[index] = val;
  }
}

// ===========================================================================
// Host-callable wrappers
// ===========================================================================

template <typename scalar_t>
void deformable_im2col_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, scalar_t* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2im_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_offset,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group, scalar_t* grad_im){

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  deformable_col2im_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2im_coord_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group,
  scalar_t* grad_offset) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  deformable_col2im_coord_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col, 
        grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}
