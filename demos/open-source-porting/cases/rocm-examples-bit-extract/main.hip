// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// \brief A simple bit extract kernel for unsigned 32-bit integer that returns an 8-bit extracted array.
/// - If the code was compiled on AMD HIP platform, the  __bitextract_u32() built-in function is used.
/// - Otherwise default C++ extract operator is used.

__global__ void bit_extract_kernel(uint32_t* d_output, const uint32_t* d_input, size_t size)
{
    const size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    const size_t stride = blockDim.x * gridDim.x;

    for(size_t i = offset; i < size; i += stride)
    {
#ifdef __HIP_PLATFORM_AMD__
        d_output[i] = __bitextract_u32(d_input[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVIDIA__ or other path */
        d_output[i] = ((d_input[i] & 0xf00) >> 8);
#endif
    }
}
