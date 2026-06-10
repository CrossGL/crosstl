#include <hip/device_functions.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>
__device__ inline float4 cgl_float4_mul(float4 lhs, float4 rhs) {
  return make_float4((lhs.x * rhs.x), (lhs.y * rhs.y), (lhs.z * rhs.z),
                     (lhs.w * rhs.w));
}

__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs) {
  return make_float4((lhs.x + rhs.x), (lhs.y + rhs.y), (lhs.z + rhs.z),
                     (lhs.w + rhs.w));
}

__device__ inline float4 cgl_scalar_mul_float4(float lhs, float4 rhs) {
  return make_float4((lhs * rhs.x), (lhs * rhs.y), (lhs * rhs.z),
                     (lhs * rhs.w));
}

__device__ inline float cgl_hip_shfl_down_sync_float(unsigned long long mask,
                                                     float value, int offset) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
  return __shfl_down_sync(mask, value, offset);
#else
  (void)mask;
  (void)offset;
  return value;
#endif
}

// CrossGL matrix value helpers
struct float2x2 {
  float m[4];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 2;
  __host__ __device__ float2x2() {}
  __host__ __device__ explicit float2x2(float diagonal) {
    for (int i = 0; i < 4; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ float2x2(float2 c0, float2 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float2x2(const Matrix& source) {
    for (int i = 0; i < 4; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float2x2(float v0, float v1, float v2, float v3) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float2x2 operator+(const float2x2& rhs) const {
    return float2x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3]);
  }
  __host__ __device__ float2x2 operator-(const float2x2& rhs) const {
    return float2x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3]);
  }
  __host__ __device__ float2x2 operator-() const {
    return float2x2(-m[0], -m[1], -m[2], -m[3]);
  }
  __host__ __device__ float2x2 operator*(float rhs) const {
    return float2x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs);
  }
  __host__ __device__ float2x2 operator/(float rhs) const {
    return float2x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs);
  }
  __host__ __device__ float2x2& operator+=(const float2x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float2x2& operator-=(const float2x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float2x2& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float2x2& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float2x2 operator*(float lhs, const float2x2& rhs) {
  return rhs * lhs;
}

struct float2x3 {
  float m[6];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 3;
  __host__ __device__ float2x3() {}
  __host__ __device__ explicit float2x3(float diagonal) {
    for (int i = 0; i < 6; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
  }
  __host__ __device__ float2x3(float3 c0, float3 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float2x3(const Matrix& source) {
    for (int i = 0; i < 6; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float2x3(float v0, float v1, float v2, float v3, float v4,
                               float v5) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float2x3 operator+(const float2x3& rhs) const {
    return float2x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5]);
  }
  __host__ __device__ float2x3 operator-(const float2x3& rhs) const {
    return float2x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5]);
  }
  __host__ __device__ float2x3 operator-() const {
    return float2x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5]);
  }
  __host__ __device__ float2x3 operator*(float rhs) const {
    return float2x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs);
  }
  __host__ __device__ float2x3 operator/(float rhs) const {
    return float2x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs);
  }
  __host__ __device__ float2x3& operator+=(const float2x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float2x3& operator-=(const float2x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float2x3& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float2x3& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float2x3 operator*(float lhs, const float2x3& rhs) {
  return rhs * lhs;
}

struct float2x4 {
  float m[8];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 4;
  __host__ __device__ float2x4() {}
  __host__ __device__ explicit float2x4(float diagonal) {
    for (int i = 0; i < 8; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
  }
  __host__ __device__ float2x4(float4 c0, float4 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float2x4(const Matrix& source) {
    for (int i = 0; i < 8; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float2x4(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float2x4 operator+(const float2x4& rhs) const {
    return float2x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7]);
  }
  __host__ __device__ float2x4 operator-(const float2x4& rhs) const {
    return float2x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7]);
  }
  __host__ __device__ float2x4 operator-() const {
    return float2x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7]);
  }
  __host__ __device__ float2x4 operator*(float rhs) const {
    return float2x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs);
  }
  __host__ __device__ float2x4 operator/(float rhs) const {
    return float2x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs);
  }
  __host__ __device__ float2x4& operator+=(const float2x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float2x4& operator-=(const float2x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float2x4& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float2x4& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float2x4 operator*(float lhs, const float2x4& rhs) {
  return rhs * lhs;
}

struct float3x2 {
  float m[6];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 2;
  __host__ __device__ float3x2() {}
  __host__ __device__ explicit float3x2(float diagonal) {
    for (int i = 0; i < 6; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ float3x2(float2 c0, float2 c1, float2 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
    m[4] = c2.x;
    m[5] = c2.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float3x2(const Matrix& source) {
    for (int i = 0; i < 6; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float3x2(float v0, float v1, float v2, float v3, float v4,
                               float v5) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float3x2 operator+(const float3x2& rhs) const {
    return float3x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5]);
  }
  __host__ __device__ float3x2 operator-(const float3x2& rhs) const {
    return float3x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5]);
  }
  __host__ __device__ float3x2 operator-() const {
    return float3x2(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5]);
  }
  __host__ __device__ float3x2 operator*(float rhs) const {
    return float3x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs);
  }
  __host__ __device__ float3x2 operator/(float rhs) const {
    return float3x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs);
  }
  __host__ __device__ float3x2& operator+=(const float3x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float3x2& operator-=(const float3x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float3x2& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float3x2& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float3x2 operator*(float lhs, const float3x2& rhs) {
  return rhs * lhs;
}

struct float3x3 {
  float m[9];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 3;
  __host__ __device__ float3x3() {}
  __host__ __device__ explicit float3x3(float diagonal) {
    for (int i = 0; i < 9; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
    m[8] = diagonal;
  }
  __host__ __device__ float3x3(float3 c0, float3 c1, float3 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
    m[6] = c2.x;
    m[7] = c2.y;
    m[8] = c2.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float3x3(const Matrix& source) {
    for (int i = 0; i < 9; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float3x3(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7, float v8) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float3x3 operator+(const float3x3& rhs) const {
    return float3x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8]);
  }
  __host__ __device__ float3x3 operator-(const float3x3& rhs) const {
    return float3x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8]);
  }
  __host__ __device__ float3x3 operator-() const {
    return float3x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                    -m[8]);
  }
  __host__ __device__ float3x3 operator*(float rhs) const {
    return float3x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs);
  }
  __host__ __device__ float3x3 operator/(float rhs) const {
    return float3x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs);
  }
  __host__ __device__ float3x3& operator+=(const float3x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float3x3& operator-=(const float3x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float3x3& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float3x3& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float3x3 operator*(float lhs, const float3x3& rhs) {
  return rhs * lhs;
}

struct float3x4 {
  float m[12];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 4;
  __host__ __device__ float3x4() {}
  __host__ __device__ explicit float3x4(float diagonal) {
    for (int i = 0; i < 12; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
    m[10] = diagonal;
  }
  __host__ __device__ float3x4(float4 c0, float4 c1, float4 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
    m[8] = c2.x;
    m[9] = c2.y;
    m[10] = c2.z;
    m[11] = c2.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float3x4(const Matrix& source) {
    for (int i = 0; i < 12; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float3x4(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7, float v8, float v9,
                               float v10, float v11) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float3x4 operator+(const float3x4& rhs) const {
    return float3x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                    m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11]);
  }
  __host__ __device__ float3x4 operator-(const float3x4& rhs) const {
    return float3x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                    m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11]);
  }
  __host__ __device__ float3x4 operator-() const {
    return float3x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                    -m[8], -m[9], -m[10], -m[11]);
  }
  __host__ __device__ float3x4 operator*(float rhs) const {
    return float3x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                    m[10] * rhs, m[11] * rhs);
  }
  __host__ __device__ float3x4 operator/(float rhs) const {
    return float3x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                    m[10] / rhs, m[11] / rhs);
  }
  __host__ __device__ float3x4& operator+=(const float3x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float3x4& operator-=(const float3x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float3x4& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float3x4& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float3x4 operator*(float lhs, const float3x4& rhs) {
  return rhs * lhs;
}

struct float4x2 {
  float m[8];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 2;
  __host__ __device__ float4x2() {}
  __host__ __device__ explicit float4x2(float diagonal) {
    for (int i = 0; i < 8; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ float4x2(float2 c0, float2 c1, float2 c2, float2 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
    m[4] = c2.x;
    m[5] = c2.y;
    m[6] = c3.x;
    m[7] = c3.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float4x2(const Matrix& source) {
    for (int i = 0; i < 8; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float4x2(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float4x2 operator+(const float4x2& rhs) const {
    return float4x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7]);
  }
  __host__ __device__ float4x2 operator-(const float4x2& rhs) const {
    return float4x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7]);
  }
  __host__ __device__ float4x2 operator-() const {
    return float4x2(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7]);
  }
  __host__ __device__ float4x2 operator*(float rhs) const {
    return float4x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs);
  }
  __host__ __device__ float4x2 operator/(float rhs) const {
    return float4x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs);
  }
  __host__ __device__ float4x2& operator+=(const float4x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float4x2& operator-=(const float4x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float4x2& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float4x2& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float4x2 operator*(float lhs, const float4x2& rhs) {
  return rhs * lhs;
}

struct float4x3 {
  float m[12];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 3;
  __host__ __device__ float4x3() {}
  __host__ __device__ explicit float4x3(float diagonal) {
    for (int i = 0; i < 12; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
    m[8] = diagonal;
  }
  __host__ __device__ float4x3(float3 c0, float3 c1, float3 c2, float3 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
    m[6] = c2.x;
    m[7] = c2.y;
    m[8] = c2.z;
    m[9] = c3.x;
    m[10] = c3.y;
    m[11] = c3.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float4x3(const Matrix& source) {
    for (int i = 0; i < 12; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float4x3(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7, float v8, float v9,
                               float v10, float v11) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float4x3 operator+(const float4x3& rhs) const {
    return float4x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                    m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11]);
  }
  __host__ __device__ float4x3 operator-(const float4x3& rhs) const {
    return float4x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                    m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11]);
  }
  __host__ __device__ float4x3 operator-() const {
    return float4x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                    -m[8], -m[9], -m[10], -m[11]);
  }
  __host__ __device__ float4x3 operator*(float rhs) const {
    return float4x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                    m[10] * rhs, m[11] * rhs);
  }
  __host__ __device__ float4x3 operator/(float rhs) const {
    return float4x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                    m[10] / rhs, m[11] / rhs);
  }
  __host__ __device__ float4x3& operator+=(const float4x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float4x3& operator-=(const float4x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float4x3& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float4x3& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float4x3 operator*(float lhs, const float4x3& rhs) {
  return rhs * lhs;
}

struct float4x4 {
  float m[16];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 4;
  __host__ __device__ float4x4() {}
  __host__ __device__ explicit float4x4(float diagonal) {
    for (int i = 0; i < 16; ++i) {
      m[i] = float(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
    m[10] = diagonal;
    m[15] = diagonal;
  }
  __host__ __device__ float4x4(float4 c0, float4 c1, float4 c2, float4 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
    m[8] = c2.x;
    m[9] = c2.y;
    m[10] = c2.z;
    m[11] = c2.w;
    m[12] = c3.x;
    m[13] = c3.y;
    m[14] = c3.z;
    m[15] = c3.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit float4x4(const Matrix& source) {
    for (int i = 0; i < 16; ++i) {
      m[i] = float(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = float(1);
        }
      }
    }
  }
  __host__ __device__ float4x4(float v0, float v1, float v2, float v3, float v4,
                               float v5, float v6, float v7, float v8, float v9,
                               float v10, float v11, float v12, float v13,
                               float v14, float v15) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
    m[12] = v12;
    m[13] = v13;
    m[14] = v14;
    m[15] = v15;
  }
  __host__ __device__ float& operator[](int index) { return m[index]; }
  __host__ __device__ const float& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ float4x4 operator+(const float4x4& rhs) const {
    return float4x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                    m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                    m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                    m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11],
                    m[12] + rhs.m[12], m[13] + rhs.m[13], m[14] + rhs.m[14],
                    m[15] + rhs.m[15]);
  }
  __host__ __device__ float4x4 operator-(const float4x4& rhs) const {
    return float4x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                    m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                    m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                    m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11],
                    m[12] - rhs.m[12], m[13] - rhs.m[13], m[14] - rhs.m[14],
                    m[15] - rhs.m[15]);
  }
  __host__ __device__ float4x4 operator-() const {
    return float4x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                    -m[8], -m[9], -m[10], -m[11], -m[12], -m[13], -m[14],
                    -m[15]);
  }
  __host__ __device__ float4x4 operator*(float rhs) const {
    return float4x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                    m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                    m[10] * rhs, m[11] * rhs, m[12] * rhs, m[13] * rhs,
                    m[14] * rhs, m[15] * rhs);
  }
  __host__ __device__ float4x4 operator/(float rhs) const {
    return float4x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                    m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                    m[10] / rhs, m[11] / rhs, m[12] / rhs, m[13] / rhs,
                    m[14] / rhs, m[15] / rhs);
  }
  __host__ __device__ float4x4& operator+=(const float4x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ float4x4& operator-=(const float4x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ float4x4& operator*=(float rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ float4x4& operator/=(float rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline float4x4 operator*(float lhs, const float4x4& rhs) {
  return rhs * lhs;
}

__host__ __device__ inline float2x2 transpose(const float2x2& value) {
  return float2x2(value.m[0], value.m[2], value.m[1], value.m[3]);
}

__host__ __device__ inline float3x2 transpose(const float2x3& value) {
  return float3x2(value.m[0], value.m[3], value.m[1], value.m[4], value.m[2],
                  value.m[5]);
}

__host__ __device__ inline float4x2 transpose(const float2x4& value) {
  return float4x2(value.m[0], value.m[4], value.m[1], value.m[5], value.m[2],
                  value.m[6], value.m[3], value.m[7]);
}

__host__ __device__ inline float2x3 transpose(const float3x2& value) {
  return float2x3(value.m[0], value.m[2], value.m[4], value.m[1], value.m[3],
                  value.m[5]);
}

__host__ __device__ inline float3x3 transpose(const float3x3& value) {
  return float3x3(value.m[0], value.m[3], value.m[6], value.m[1], value.m[4],
                  value.m[7], value.m[2], value.m[5], value.m[8]);
}

__host__ __device__ inline float4x3 transpose(const float3x4& value) {
  return float4x3(value.m[0], value.m[4], value.m[8], value.m[1], value.m[5],
                  value.m[9], value.m[2], value.m[6], value.m[10], value.m[3],
                  value.m[7], value.m[11]);
}

__host__ __device__ inline float2x4 transpose(const float4x2& value) {
  return float2x4(value.m[0], value.m[2], value.m[4], value.m[6], value.m[1],
                  value.m[3], value.m[5], value.m[7]);
}

__host__ __device__ inline float3x4 transpose(const float4x3& value) {
  return float3x4(value.m[0], value.m[3], value.m[6], value.m[9], value.m[1],
                  value.m[4], value.m[7], value.m[10], value.m[2], value.m[5],
                  value.m[8], value.m[11]);
}

__host__ __device__ inline float4x4 transpose(const float4x4& value) {
  return float4x4(value.m[0], value.m[4], value.m[8], value.m[12], value.m[1],
                  value.m[5], value.m[9], value.m[13], value.m[2], value.m[6],
                  value.m[10], value.m[14], value.m[3], value.m[7], value.m[11],
                  value.m[15]);
}

__host__ __device__ inline float2x2 inverse(const float2x2& value) {
  float2x2 a(value);
  float2x2 result(float(1));
  for (int column = 0; column < 2; ++column) {
    int pivot_row = column;
    float pivot_value = a.m[column * 2 + column];
    float pivot_abs = pivot_value < float(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 2; ++row) {
      float candidate_value = a.m[column * 2 + row];
      float candidate_abs =
          candidate_value < float(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == float(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 2; ++c) {
        float tmp = a.m[c * 2 + column];
        a.m[c * 2 + column] = a.m[c * 2 + pivot_row];
        a.m[c * 2 + pivot_row] = tmp;
        tmp = result.m[c * 2 + column];
        result.m[c * 2 + column] = result.m[c * 2 + pivot_row];
        result.m[c * 2 + pivot_row] = tmp;
      }
    }
    float pivot = a.m[column * 2 + column];
    for (int c = 0; c < 2; ++c) {
      a.m[c * 2 + column] /= pivot;
      result.m[c * 2 + column] /= pivot;
    }
    for (int row = 0; row < 2; ++row) {
      if (row == column) {
        continue;
      }
      float factor = a.m[column * 2 + row];
      for (int c = 0; c < 2; ++c) {
        a.m[c * 2 + row] -= factor * a.m[c * 2 + column];
        result.m[c * 2 + row] -= factor * result.m[c * 2 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline float3x3 inverse(const float3x3& value) {
  float3x3 a(value);
  float3x3 result(float(1));
  for (int column = 0; column < 3; ++column) {
    int pivot_row = column;
    float pivot_value = a.m[column * 3 + column];
    float pivot_abs = pivot_value < float(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 3; ++row) {
      float candidate_value = a.m[column * 3 + row];
      float candidate_abs =
          candidate_value < float(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == float(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 3; ++c) {
        float tmp = a.m[c * 3 + column];
        a.m[c * 3 + column] = a.m[c * 3 + pivot_row];
        a.m[c * 3 + pivot_row] = tmp;
        tmp = result.m[c * 3 + column];
        result.m[c * 3 + column] = result.m[c * 3 + pivot_row];
        result.m[c * 3 + pivot_row] = tmp;
      }
    }
    float pivot = a.m[column * 3 + column];
    for (int c = 0; c < 3; ++c) {
      a.m[c * 3 + column] /= pivot;
      result.m[c * 3 + column] /= pivot;
    }
    for (int row = 0; row < 3; ++row) {
      if (row == column) {
        continue;
      }
      float factor = a.m[column * 3 + row];
      for (int c = 0; c < 3; ++c) {
        a.m[c * 3 + row] -= factor * a.m[c * 3 + column];
        result.m[c * 3 + row] -= factor * result.m[c * 3 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline float4x4 inverse(const float4x4& value) {
  float4x4 a(value);
  float4x4 result(float(1));
  for (int column = 0; column < 4; ++column) {
    int pivot_row = column;
    float pivot_value = a.m[column * 4 + column];
    float pivot_abs = pivot_value < float(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 4; ++row) {
      float candidate_value = a.m[column * 4 + row];
      float candidate_abs =
          candidate_value < float(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == float(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 4; ++c) {
        float tmp = a.m[c * 4 + column];
        a.m[c * 4 + column] = a.m[c * 4 + pivot_row];
        a.m[c * 4 + pivot_row] = tmp;
        tmp = result.m[c * 4 + column];
        result.m[c * 4 + column] = result.m[c * 4 + pivot_row];
        result.m[c * 4 + pivot_row] = tmp;
      }
    }
    float pivot = a.m[column * 4 + column];
    for (int c = 0; c < 4; ++c) {
      a.m[c * 4 + column] /= pivot;
      result.m[c * 4 + column] /= pivot;
    }
    for (int row = 0; row < 4; ++row) {
      if (row == column) {
        continue;
      }
      float factor = a.m[column * 4 + row];
      for (int c = 0; c < 4; ++c) {
        a.m[c * 4 + row] -= factor * a.m[c * 4 + column];
        result.m[c * 4 + row] -= factor * result.m[c * 4 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline float2 operator*(const float2x2& lhs,
                                            const float2& rhs) {
  return make_float2(lhs.m[0] * rhs.x + lhs.m[2] * rhs.y,
                     lhs.m[1] * rhs.x + lhs.m[3] * rhs.y);
}

__host__ __device__ inline float2 operator*(const float2& lhs,
                                            const float2x2& rhs) {
  return make_float2(lhs.x * rhs.m[0] + lhs.y * rhs.m[1],
                     lhs.x * rhs.m[2] + lhs.y * rhs.m[3]);
}

__host__ __device__ inline float2x2 operator*(const float2x2& lhs,
                                              const float2x2& rhs) {
  return float2x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3]);
}

__host__ __device__ inline float3x2 operator*(const float2x2& lhs,
                                              const float3x2& rhs) {
  return float3x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5]);
}

__host__ __device__ inline float4x2 operator*(const float2x2& lhs,
                                              const float4x2& rhs) {
  return float4x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                  lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7],
                  lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7]);
}

__host__ __device__ inline float3 operator*(const float2x3& lhs,
                                            const float2& rhs) {
  return make_float3(lhs.m[0] * rhs.x + lhs.m[3] * rhs.y,
                     lhs.m[1] * rhs.x + lhs.m[4] * rhs.y,
                     lhs.m[2] * rhs.x + lhs.m[5] * rhs.y);
}

__host__ __device__ inline float2 operator*(const float3& lhs,
                                            const float2x3& rhs) {
  return make_float2(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                     lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5]);
}

__host__ __device__ inline float2x3 operator*(const float2x3& lhs,
                                              const float2x2& rhs) {
  return float2x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3]);
}

__host__ __device__ inline float3x3 operator*(const float2x3& lhs,
                                              const float3x2& rhs) {
  return float3x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                  lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5]);
}

__host__ __device__ inline float4x3 operator*(const float2x3& lhs,
                                              const float4x2& rhs) {
  return float4x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                  lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                  lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7],
                  lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7],
                  lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7]);
}

__host__ __device__ inline float4 operator*(const float2x4& lhs,
                                            const float2& rhs) {
  return make_float4(
      lhs.m[0] * rhs.x + lhs.m[4] * rhs.y, lhs.m[1] * rhs.x + lhs.m[5] * rhs.y,
      lhs.m[2] * rhs.x + lhs.m[6] * rhs.y, lhs.m[3] * rhs.x + lhs.m[7] * rhs.y);
}

__host__ __device__ inline float2 operator*(const float4& lhs,
                                            const float2x4& rhs) {
  return make_float2(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] +
          lhs.w * rhs.m[7]);
}

__host__ __device__ inline float2x4 operator*(const float2x4& lhs,
                                              const float2x2& rhs) {
  return float2x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3]);
}

__host__ __device__ inline float3x4 operator*(const float2x4& lhs,
                                              const float3x2& rhs) {
  return float3x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                  lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5],
                  lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5]);
}

__host__ __device__ inline float4x4 operator*(const float2x4& lhs,
                                              const float4x2& rhs) {
  return float4x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                  lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                  lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                  lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                  lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                  lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5],
                  lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5],
                  lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7],
                  lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7],
                  lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                  lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7]);
}

__host__ __device__ inline float2 operator*(const float3x2& lhs,
                                            const float3& rhs) {
  return make_float2(lhs.m[0] * rhs.x + lhs.m[2] * rhs.y + lhs.m[4] * rhs.z,
                     lhs.m[1] * rhs.x + lhs.m[3] * rhs.y + lhs.m[5] * rhs.z);
}

__host__ __device__ inline float3 operator*(const float2& lhs,
                                            const float3x2& rhs) {
  return make_float3(lhs.x * rhs.m[0] + lhs.y * rhs.m[1],
                     lhs.x * rhs.m[2] + lhs.y * rhs.m[3],
                     lhs.x * rhs.m[4] + lhs.y * rhs.m[5]);
}

__host__ __device__ inline float2x2 operator*(const float3x2& lhs,
                                              const float2x3& rhs) {
  return float2x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5]);
}

__host__ __device__ inline float3x2 operator*(const float3x2& lhs,
                                              const float3x3& rhs) {
  return float3x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7] + lhs.m[4] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[5] * rhs.m[8]);
}

__host__ __device__ inline float4x2 operator*(const float3x2& lhs,
                                              const float4x3& rhs) {
  return float4x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7] + lhs.m[4] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[5] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[2] * rhs.m[10] + lhs.m[4] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[3] * rhs.m[10] + lhs.m[5] * rhs.m[11]);
}

__host__ __device__ inline float3 operator*(const float3x3& lhs,
                                            const float3& rhs) {
  return make_float3(lhs.m[0] * rhs.x + lhs.m[3] * rhs.y + lhs.m[6] * rhs.z,
                     lhs.m[1] * rhs.x + lhs.m[4] * rhs.y + lhs.m[7] * rhs.z,
                     lhs.m[2] * rhs.x + lhs.m[5] * rhs.y + lhs.m[8] * rhs.z);
}

__host__ __device__ inline float3 operator*(const float3& lhs,
                                            const float3x3& rhs) {
  return make_float3(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                     lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5],
                     lhs.x * rhs.m[6] + lhs.y * rhs.m[7] + lhs.z * rhs.m[8]);
}

__host__ __device__ inline float2x3 operator*(const float3x3& lhs,
                                              const float2x3& rhs) {
  return float2x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5]);
}

__host__ __device__ inline float3x3 operator*(const float3x3& lhs,
                                              const float3x3& rhs) {
  return float3x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[6] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[7] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[8] * rhs.m[8]);
}

__host__ __device__ inline float4x3 operator*(const float3x3& lhs,
                                              const float4x3& rhs) {
  return float4x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[6] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[7] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[3] * rhs.m[10] + lhs.m[6] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[4] * rhs.m[10] + lhs.m[7] * rhs.m[11],
      lhs.m[2] * rhs.m[9] + lhs.m[5] * rhs.m[10] + lhs.m[8] * rhs.m[11]);
}

__host__ __device__ inline float4 operator*(const float3x4& lhs,
                                            const float3& rhs) {
  return make_float4(lhs.m[0] * rhs.x + lhs.m[4] * rhs.y + lhs.m[8] * rhs.z,
                     lhs.m[1] * rhs.x + lhs.m[5] * rhs.y + lhs.m[9] * rhs.z,
                     lhs.m[2] * rhs.x + lhs.m[6] * rhs.y + lhs.m[10] * rhs.z,
                     lhs.m[3] * rhs.x + lhs.m[7] * rhs.y + lhs.m[11] * rhs.z);
}

__host__ __device__ inline float3 operator*(const float4& lhs,
                                            const float3x4& rhs) {
  return make_float3(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] + lhs.w * rhs.m[7],
      lhs.x * rhs.m[8] + lhs.y * rhs.m[9] + lhs.z * rhs.m[10] +
          lhs.w * rhs.m[11]);
}

__host__ __device__ inline float2x4 operator*(const float3x4& lhs,
                                              const float2x3& rhs) {
  return float2x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5]);
}

__host__ __device__ inline float3x4 operator*(const float3x4& lhs,
                                              const float3x3& rhs) {
  return float3x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[9] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7] + lhs.m[10] * rhs.m[8],
      lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7] + lhs.m[11] * rhs.m[8]);
}

__host__ __device__ inline float4x4 operator*(const float3x4& lhs,
                                              const float4x3& rhs) {
  return float4x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[9] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7] + lhs.m[10] * rhs.m[8],
      lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7] + lhs.m[11] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[4] * rhs.m[10] + lhs.m[8] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[5] * rhs.m[10] + lhs.m[9] * rhs.m[11],
      lhs.m[2] * rhs.m[9] + lhs.m[6] * rhs.m[10] + lhs.m[10] * rhs.m[11],
      lhs.m[3] * rhs.m[9] + lhs.m[7] * rhs.m[10] + lhs.m[11] * rhs.m[11]);
}

__host__ __device__ inline float2 operator*(const float4x2& lhs,
                                            const float4& rhs) {
  return make_float2(
      lhs.m[0] * rhs.x + lhs.m[2] * rhs.y + lhs.m[4] * rhs.z + lhs.m[6] * rhs.w,
      lhs.m[1] * rhs.x + lhs.m[3] * rhs.y + lhs.m[5] * rhs.z +
          lhs.m[7] * rhs.w);
}

__host__ __device__ inline float4 operator*(const float2& lhs,
                                            const float4x2& rhs) {
  return make_float4(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1], lhs.x * rhs.m[2] + lhs.y * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5], lhs.x * rhs.m[6] + lhs.y * rhs.m[7]);
}

__host__ __device__ inline float2x2 operator*(const float4x2& lhs,
                                              const float2x4& rhs) {
  return float2x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                      lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                      lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7]);
}

__host__ __device__ inline float3x2 operator*(const float4x2& lhs,
                                              const float3x4& rhs) {
  return float3x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                      lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                      lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[2] * rhs.m[9] +
                      lhs.m[4] * rhs.m[10] + lhs.m[6] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                      lhs.m[5] * rhs.m[10] + lhs.m[7] * rhs.m[11]);
}

__host__ __device__ inline float4x2 operator*(const float4x2& lhs,
                                              const float4x4& rhs) {
  return float4x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                      lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                      lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[2] * rhs.m[9] +
                      lhs.m[4] * rhs.m[10] + lhs.m[6] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                      lhs.m[5] * rhs.m[10] + lhs.m[7] * rhs.m[11],
                  lhs.m[0] * rhs.m[12] + lhs.m[2] * rhs.m[13] +
                      lhs.m[4] * rhs.m[14] + lhs.m[6] * rhs.m[15],
                  lhs.m[1] * rhs.m[12] + lhs.m[3] * rhs.m[13] +
                      lhs.m[5] * rhs.m[14] + lhs.m[7] * rhs.m[15]);
}

__host__ __device__ inline float3 operator*(const float4x3& lhs,
                                            const float4& rhs) {
  return make_float3(
      lhs.m[0] * rhs.x + lhs.m[3] * rhs.y + lhs.m[6] * rhs.z + lhs.m[9] * rhs.w,
      lhs.m[1] * rhs.x + lhs.m[4] * rhs.y + lhs.m[7] * rhs.z +
          lhs.m[10] * rhs.w,
      lhs.m[2] * rhs.x + lhs.m[5] * rhs.y + lhs.m[8] * rhs.z +
          lhs.m[11] * rhs.w);
}

__host__ __device__ inline float4 operator*(const float3& lhs,
                                            const float4x3& rhs) {
  return make_float4(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                     lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5],
                     lhs.x * rhs.m[6] + lhs.y * rhs.m[7] + lhs.z * rhs.m[8],
                     lhs.x * rhs.m[9] + lhs.y * rhs.m[10] + lhs.z * rhs.m[11]);
}

__host__ __device__ inline float2x3 operator*(const float4x3& lhs,
                                              const float2x4& rhs) {
  return float2x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7]);
}

__host__ __device__ inline float3x3 operator*(const float4x3& lhs,
                                              const float3x4& rhs) {
  return float3x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                      lhs.m[6] * rhs.m[10] + lhs.m[9] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                      lhs.m[7] * rhs.m[10] + lhs.m[10] * rhs.m[11],
                  lhs.m[2] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                      lhs.m[8] * rhs.m[10] + lhs.m[11] * rhs.m[11]);
}

__host__ __device__ inline float4x3 operator*(const float4x3& lhs,
                                              const float4x4& rhs) {
  return float4x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                      lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                      lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                      lhs.m[6] * rhs.m[10] + lhs.m[9] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                      lhs.m[7] * rhs.m[10] + lhs.m[10] * rhs.m[11],
                  lhs.m[2] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                      lhs.m[8] * rhs.m[10] + lhs.m[11] * rhs.m[11],
                  lhs.m[0] * rhs.m[12] + lhs.m[3] * rhs.m[13] +
                      lhs.m[6] * rhs.m[14] + lhs.m[9] * rhs.m[15],
                  lhs.m[1] * rhs.m[12] + lhs.m[4] * rhs.m[13] +
                      lhs.m[7] * rhs.m[14] + lhs.m[10] * rhs.m[15],
                  lhs.m[2] * rhs.m[12] + lhs.m[5] * rhs.m[13] +
                      lhs.m[8] * rhs.m[14] + lhs.m[11] * rhs.m[15]);
}

__host__ __device__ inline float4 operator*(const float4x4& lhs,
                                            const float4& rhs) {
  return make_float4(lhs.m[0] * rhs.x + lhs.m[4] * rhs.y + lhs.m[8] * rhs.z +
                         lhs.m[12] * rhs.w,
                     lhs.m[1] * rhs.x + lhs.m[5] * rhs.y + lhs.m[9] * rhs.z +
                         lhs.m[13] * rhs.w,
                     lhs.m[2] * rhs.x + lhs.m[6] * rhs.y + lhs.m[10] * rhs.z +
                         lhs.m[14] * rhs.w,
                     lhs.m[3] * rhs.x + lhs.m[7] * rhs.y + lhs.m[11] * rhs.z +
                         lhs.m[15] * rhs.w);
}

__host__ __device__ inline float4 operator*(const float4& lhs,
                                            const float4x4& rhs) {
  return make_float4(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] + lhs.w * rhs.m[7],
      lhs.x * rhs.m[8] + lhs.y * rhs.m[9] + lhs.z * rhs.m[10] +
          lhs.w * rhs.m[11],
      lhs.x * rhs.m[12] + lhs.y * rhs.m[13] + lhs.z * rhs.m[14] +
          lhs.w * rhs.m[15]);
}

__host__ __device__ inline float2x4 operator*(const float4x4& lhs,
                                              const float2x4& rhs) {
  return float2x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                      lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                      lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                      lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                  lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                      lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7]);
}

__host__ __device__ inline float3x4 operator*(const float4x4& lhs,
                                              const float3x4& rhs) {
  return float3x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                      lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                      lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                      lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                  lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                      lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                      lhs.m[8] * rhs.m[10] + lhs.m[12] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                      lhs.m[9] * rhs.m[10] + lhs.m[13] * rhs.m[11],
                  lhs.m[2] * rhs.m[8] + lhs.m[6] * rhs.m[9] +
                      lhs.m[10] * rhs.m[10] + lhs.m[14] * rhs.m[11],
                  lhs.m[3] * rhs.m[8] + lhs.m[7] * rhs.m[9] +
                      lhs.m[11] * rhs.m[10] + lhs.m[15] * rhs.m[11]);
}

__host__ __device__ inline float4x4 operator*(const float4x4& lhs,
                                              const float4x4& rhs) {
  return float4x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                      lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                  lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                      lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                  lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                      lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                  lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                      lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                  lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                      lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                  lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                      lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                  lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                      lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                  lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                      lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7],
                  lhs.m[0] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                      lhs.m[8] * rhs.m[10] + lhs.m[12] * rhs.m[11],
                  lhs.m[1] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                      lhs.m[9] * rhs.m[10] + lhs.m[13] * rhs.m[11],
                  lhs.m[2] * rhs.m[8] + lhs.m[6] * rhs.m[9] +
                      lhs.m[10] * rhs.m[10] + lhs.m[14] * rhs.m[11],
                  lhs.m[3] * rhs.m[8] + lhs.m[7] * rhs.m[9] +
                      lhs.m[11] * rhs.m[10] + lhs.m[15] * rhs.m[11],
                  lhs.m[0] * rhs.m[12] + lhs.m[4] * rhs.m[13] +
                      lhs.m[8] * rhs.m[14] + lhs.m[12] * rhs.m[15],
                  lhs.m[1] * rhs.m[12] + lhs.m[5] * rhs.m[13] +
                      lhs.m[9] * rhs.m[14] + lhs.m[13] * rhs.m[15],
                  lhs.m[2] * rhs.m[12] + lhs.m[6] * rhs.m[13] +
                      lhs.m[10] * rhs.m[14] + lhs.m[14] * rhs.m[15],
                  lhs.m[3] * rhs.m[12] + lhs.m[7] * rhs.m[13] +
                      lhs.m[11] * rhs.m[14] + lhs.m[15] * rhs.m[15]);
}

struct double2x2 {
  double m[4];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 2;
  __host__ __device__ double2x2() {}
  __host__ __device__ explicit double2x2(double diagonal) {
    for (int i = 0; i < 4; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ double2x2(double2 c0, double2 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double2x2(const Matrix& source) {
    for (int i = 0; i < 4; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double2x2(double v0, double v1, double v2, double v3) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double2x2 operator+(const double2x2& rhs) const {
    return double2x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3]);
  }
  __host__ __device__ double2x2 operator-(const double2x2& rhs) const {
    return double2x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3]);
  }
  __host__ __device__ double2x2 operator-() const {
    return double2x2(-m[0], -m[1], -m[2], -m[3]);
  }
  __host__ __device__ double2x2 operator*(double rhs) const {
    return double2x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs);
  }
  __host__ __device__ double2x2 operator/(double rhs) const {
    return double2x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs);
  }
  __host__ __device__ double2x2& operator+=(const double2x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double2x2& operator-=(const double2x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double2x2& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double2x2& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double2x2 operator*(double lhs,
                                               const double2x2& rhs) {
  return rhs * lhs;
}

struct double2x3 {
  double m[6];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 3;
  __host__ __device__ double2x3() {}
  __host__ __device__ explicit double2x3(double diagonal) {
    for (int i = 0; i < 6; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
  }
  __host__ __device__ double2x3(double3 c0, double3 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double2x3(const Matrix& source) {
    for (int i = 0; i < 6; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double2x3(double v0, double v1, double v2, double v3,
                                double v4, double v5) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double2x3 operator+(const double2x3& rhs) const {
    return double2x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5]);
  }
  __host__ __device__ double2x3 operator-(const double2x3& rhs) const {
    return double2x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5]);
  }
  __host__ __device__ double2x3 operator-() const {
    return double2x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5]);
  }
  __host__ __device__ double2x3 operator*(double rhs) const {
    return double2x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs);
  }
  __host__ __device__ double2x3 operator/(double rhs) const {
    return double2x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs);
  }
  __host__ __device__ double2x3& operator+=(const double2x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double2x3& operator-=(const double2x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double2x3& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double2x3& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double2x3 operator*(double lhs,
                                               const double2x3& rhs) {
  return rhs * lhs;
}

struct double2x4 {
  double m[8];
  static const int CGL_COLUMNS = 2;
  static const int CGL_ROWS = 4;
  __host__ __device__ double2x4() {}
  __host__ __device__ explicit double2x4(double diagonal) {
    for (int i = 0; i < 8; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
  }
  __host__ __device__ double2x4(double4 c0, double4 c1) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double2x4(const Matrix& source) {
    for (int i = 0; i < 8; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double2x4(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double2x4 operator+(const double2x4& rhs) const {
    return double2x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7]);
  }
  __host__ __device__ double2x4 operator-(const double2x4& rhs) const {
    return double2x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7]);
  }
  __host__ __device__ double2x4 operator-() const {
    return double2x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7]);
  }
  __host__ __device__ double2x4 operator*(double rhs) const {
    return double2x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs);
  }
  __host__ __device__ double2x4 operator/(double rhs) const {
    return double2x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs);
  }
  __host__ __device__ double2x4& operator+=(const double2x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double2x4& operator-=(const double2x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double2x4& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double2x4& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double2x4 operator*(double lhs,
                                               const double2x4& rhs) {
  return rhs * lhs;
}

struct double3x2 {
  double m[6];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 2;
  __host__ __device__ double3x2() {}
  __host__ __device__ explicit double3x2(double diagonal) {
    for (int i = 0; i < 6; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ double3x2(double2 c0, double2 c1, double2 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
    m[4] = c2.x;
    m[5] = c2.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double3x2(const Matrix& source) {
    for (int i = 0; i < 6; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double3x2(double v0, double v1, double v2, double v3,
                                double v4, double v5) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double3x2 operator+(const double3x2& rhs) const {
    return double3x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5]);
  }
  __host__ __device__ double3x2 operator-(const double3x2& rhs) const {
    return double3x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5]);
  }
  __host__ __device__ double3x2 operator-() const {
    return double3x2(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5]);
  }
  __host__ __device__ double3x2 operator*(double rhs) const {
    return double3x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs);
  }
  __host__ __device__ double3x2 operator/(double rhs) const {
    return double3x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs);
  }
  __host__ __device__ double3x2& operator+=(const double3x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double3x2& operator-=(const double3x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double3x2& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double3x2& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double3x2 operator*(double lhs,
                                               const double3x2& rhs) {
  return rhs * lhs;
}

struct double3x3 {
  double m[9];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 3;
  __host__ __device__ double3x3() {}
  __host__ __device__ explicit double3x3(double diagonal) {
    for (int i = 0; i < 9; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
    m[8] = diagonal;
  }
  __host__ __device__ double3x3(double3 c0, double3 c1, double3 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
    m[6] = c2.x;
    m[7] = c2.y;
    m[8] = c2.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double3x3(const Matrix& source) {
    for (int i = 0; i < 9; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double3x3(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7,
                                double v8) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double3x3 operator+(const double3x3& rhs) const {
    return double3x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8]);
  }
  __host__ __device__ double3x3 operator-(const double3x3& rhs) const {
    return double3x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8]);
  }
  __host__ __device__ double3x3 operator-() const {
    return double3x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                     -m[8]);
  }
  __host__ __device__ double3x3 operator*(double rhs) const {
    return double3x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs);
  }
  __host__ __device__ double3x3 operator/(double rhs) const {
    return double3x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs);
  }
  __host__ __device__ double3x3& operator+=(const double3x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double3x3& operator-=(const double3x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double3x3& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double3x3& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double3x3 operator*(double lhs,
                                               const double3x3& rhs) {
  return rhs * lhs;
}

struct double3x4 {
  double m[12];
  static const int CGL_COLUMNS = 3;
  static const int CGL_ROWS = 4;
  __host__ __device__ double3x4() {}
  __host__ __device__ explicit double3x4(double diagonal) {
    for (int i = 0; i < 12; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
    m[10] = diagonal;
  }
  __host__ __device__ double3x4(double4 c0, double4 c1, double4 c2) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
    m[8] = c2.x;
    m[9] = c2.y;
    m[10] = c2.z;
    m[11] = c2.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double3x4(const Matrix& source) {
    for (int i = 0; i < 12; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 3; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double3x4(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7,
                                double v8, double v9, double v10, double v11) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double3x4 operator+(const double3x4& rhs) const {
    return double3x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                     m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11]);
  }
  __host__ __device__ double3x4 operator-(const double3x4& rhs) const {
    return double3x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                     m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11]);
  }
  __host__ __device__ double3x4 operator-() const {
    return double3x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                     -m[8], -m[9], -m[10], -m[11]);
  }
  __host__ __device__ double3x4 operator*(double rhs) const {
    return double3x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                     m[10] * rhs, m[11] * rhs);
  }
  __host__ __device__ double3x4 operator/(double rhs) const {
    return double3x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                     m[10] / rhs, m[11] / rhs);
  }
  __host__ __device__ double3x4& operator+=(const double3x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double3x4& operator-=(const double3x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double3x4& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double3x4& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double3x4 operator*(double lhs,
                                               const double3x4& rhs) {
  return rhs * lhs;
}

struct double4x2 {
  double m[8];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 2;
  __host__ __device__ double4x2() {}
  __host__ __device__ explicit double4x2(double diagonal) {
    for (int i = 0; i < 8; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[3] = diagonal;
  }
  __host__ __device__ double4x2(double2 c0, double2 c1, double2 c2,
                                double2 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c1.x;
    m[3] = c1.y;
    m[4] = c2.x;
    m[5] = c2.y;
    m[6] = c3.x;
    m[7] = c3.y;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double4x2(const Matrix& source) {
    for (int i = 0; i < 8; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 2; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 2 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 2 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double4x2(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double4x2 operator+(const double4x2& rhs) const {
    return double4x2(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7]);
  }
  __host__ __device__ double4x2 operator-(const double4x2& rhs) const {
    return double4x2(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7]);
  }
  __host__ __device__ double4x2 operator-() const {
    return double4x2(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7]);
  }
  __host__ __device__ double4x2 operator*(double rhs) const {
    return double4x2(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs);
  }
  __host__ __device__ double4x2 operator/(double rhs) const {
    return double4x2(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs);
  }
  __host__ __device__ double4x2& operator+=(const double4x2& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double4x2& operator-=(const double4x2& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double4x2& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double4x2& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double4x2 operator*(double lhs,
                                               const double4x2& rhs) {
  return rhs * lhs;
}

struct double4x3 {
  double m[12];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 3;
  __host__ __device__ double4x3() {}
  __host__ __device__ explicit double4x3(double diagonal) {
    for (int i = 0; i < 12; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[4] = diagonal;
    m[8] = diagonal;
  }
  __host__ __device__ double4x3(double3 c0, double3 c1, double3 c2,
                                double3 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c1.x;
    m[4] = c1.y;
    m[5] = c1.z;
    m[6] = c2.x;
    m[7] = c2.y;
    m[8] = c2.z;
    m[9] = c3.x;
    m[10] = c3.y;
    m[11] = c3.z;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double4x3(const Matrix& source) {
    for (int i = 0; i < 12; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 3; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 3 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 3 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double4x3(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7,
                                double v8, double v9, double v10, double v11) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double4x3 operator+(const double4x3& rhs) const {
    return double4x3(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                     m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11]);
  }
  __host__ __device__ double4x3 operator-(const double4x3& rhs) const {
    return double4x3(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                     m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11]);
  }
  __host__ __device__ double4x3 operator-() const {
    return double4x3(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                     -m[8], -m[9], -m[10], -m[11]);
  }
  __host__ __device__ double4x3 operator*(double rhs) const {
    return double4x3(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                     m[10] * rhs, m[11] * rhs);
  }
  __host__ __device__ double4x3 operator/(double rhs) const {
    return double4x3(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                     m[10] / rhs, m[11] / rhs);
  }
  __host__ __device__ double4x3& operator+=(const double4x3& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double4x3& operator-=(const double4x3& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double4x3& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double4x3& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double4x3 operator*(double lhs,
                                               const double4x3& rhs) {
  return rhs * lhs;
}

struct double4x4 {
  double m[16];
  static const int CGL_COLUMNS = 4;
  static const int CGL_ROWS = 4;
  __host__ __device__ double4x4() {}
  __host__ __device__ explicit double4x4(double diagonal) {
    for (int i = 0; i < 16; ++i) {
      m[i] = double(0);
    }
    m[0] = diagonal;
    m[5] = diagonal;
    m[10] = diagonal;
    m[15] = diagonal;
  }
  __host__ __device__ double4x4(double4 c0, double4 c1, double4 c2,
                                double4 c3) {
    m[0] = c0.x;
    m[1] = c0.y;
    m[2] = c0.z;
    m[3] = c0.w;
    m[4] = c1.x;
    m[5] = c1.y;
    m[6] = c1.z;
    m[7] = c1.w;
    m[8] = c2.x;
    m[9] = c2.y;
    m[10] = c2.z;
    m[11] = c2.w;
    m[12] = c3.x;
    m[13] = c3.y;
    m[14] = c3.z;
    m[15] = c3.w;
  }
  template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS),
            typename = decltype(Matrix::CGL_ROWS)>
  __host__ __device__ explicit double4x4(const Matrix& source) {
    for (int i = 0; i < 16; ++i) {
      m[i] = double(0);
    }
    for (int column = 0; column < 4; ++column) {
      for (int row = 0; row < 4; ++row) {
        if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {
          m[column * 4 + row] = source.m[column * Matrix::CGL_ROWS + row];
        } else if (column == row) {
          m[column * 4 + row] = double(1);
        }
      }
    }
  }
  __host__ __device__ double4x4(double v0, double v1, double v2, double v3,
                                double v4, double v5, double v6, double v7,
                                double v8, double v9, double v10, double v11,
                                double v12, double v13, double v14,
                                double v15) {
    m[0] = v0;
    m[1] = v1;
    m[2] = v2;
    m[3] = v3;
    m[4] = v4;
    m[5] = v5;
    m[6] = v6;
    m[7] = v7;
    m[8] = v8;
    m[9] = v9;
    m[10] = v10;
    m[11] = v11;
    m[12] = v12;
    m[13] = v13;
    m[14] = v14;
    m[15] = v15;
  }
  __host__ __device__ double& operator[](int index) { return m[index]; }
  __host__ __device__ const double& operator[](int index) const {
    return m[index];
  }
  __host__ __device__ double4x4 operator+(const double4x4& rhs) const {
    return double4x4(m[0] + rhs.m[0], m[1] + rhs.m[1], m[2] + rhs.m[2],
                     m[3] + rhs.m[3], m[4] + rhs.m[4], m[5] + rhs.m[5],
                     m[6] + rhs.m[6], m[7] + rhs.m[7], m[8] + rhs.m[8],
                     m[9] + rhs.m[9], m[10] + rhs.m[10], m[11] + rhs.m[11],
                     m[12] + rhs.m[12], m[13] + rhs.m[13], m[14] + rhs.m[14],
                     m[15] + rhs.m[15]);
  }
  __host__ __device__ double4x4 operator-(const double4x4& rhs) const {
    return double4x4(m[0] - rhs.m[0], m[1] - rhs.m[1], m[2] - rhs.m[2],
                     m[3] - rhs.m[3], m[4] - rhs.m[4], m[5] - rhs.m[5],
                     m[6] - rhs.m[6], m[7] - rhs.m[7], m[8] - rhs.m[8],
                     m[9] - rhs.m[9], m[10] - rhs.m[10], m[11] - rhs.m[11],
                     m[12] - rhs.m[12], m[13] - rhs.m[13], m[14] - rhs.m[14],
                     m[15] - rhs.m[15]);
  }
  __host__ __device__ double4x4 operator-() const {
    return double4x4(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], -m[7],
                     -m[8], -m[9], -m[10], -m[11], -m[12], -m[13], -m[14],
                     -m[15]);
  }
  __host__ __device__ double4x4 operator*(double rhs) const {
    return double4x4(m[0] * rhs, m[1] * rhs, m[2] * rhs, m[3] * rhs, m[4] * rhs,
                     m[5] * rhs, m[6] * rhs, m[7] * rhs, m[8] * rhs, m[9] * rhs,
                     m[10] * rhs, m[11] * rhs, m[12] * rhs, m[13] * rhs,
                     m[14] * rhs, m[15] * rhs);
  }
  __host__ __device__ double4x4 operator/(double rhs) const {
    return double4x4(m[0] / rhs, m[1] / rhs, m[2] / rhs, m[3] / rhs, m[4] / rhs,
                     m[5] / rhs, m[6] / rhs, m[7] / rhs, m[8] / rhs, m[9] / rhs,
                     m[10] / rhs, m[11] / rhs, m[12] / rhs, m[13] / rhs,
                     m[14] / rhs, m[15] / rhs);
  }
  __host__ __device__ double4x4& operator+=(const double4x4& rhs) {
    *this = *this + rhs;
    return *this;
  }
  __host__ __device__ double4x4& operator-=(const double4x4& rhs) {
    *this = *this - rhs;
    return *this;
  }
  __host__ __device__ double4x4& operator*=(double rhs) {
    *this = *this * rhs;
    return *this;
  }
  __host__ __device__ double4x4& operator/=(double rhs) {
    *this = *this / rhs;
    return *this;
  }
};

__host__ __device__ inline double4x4 operator*(double lhs,
                                               const double4x4& rhs) {
  return rhs * lhs;
}

__host__ __device__ inline double2x2 transpose(const double2x2& value) {
  return double2x2(value.m[0], value.m[2], value.m[1], value.m[3]);
}

__host__ __device__ inline double3x2 transpose(const double2x3& value) {
  return double3x2(value.m[0], value.m[3], value.m[1], value.m[4], value.m[2],
                   value.m[5]);
}

__host__ __device__ inline double4x2 transpose(const double2x4& value) {
  return double4x2(value.m[0], value.m[4], value.m[1], value.m[5], value.m[2],
                   value.m[6], value.m[3], value.m[7]);
}

__host__ __device__ inline double2x3 transpose(const double3x2& value) {
  return double2x3(value.m[0], value.m[2], value.m[4], value.m[1], value.m[3],
                   value.m[5]);
}

__host__ __device__ inline double3x3 transpose(const double3x3& value) {
  return double3x3(value.m[0], value.m[3], value.m[6], value.m[1], value.m[4],
                   value.m[7], value.m[2], value.m[5], value.m[8]);
}

__host__ __device__ inline double4x3 transpose(const double3x4& value) {
  return double4x3(value.m[0], value.m[4], value.m[8], value.m[1], value.m[5],
                   value.m[9], value.m[2], value.m[6], value.m[10], value.m[3],
                   value.m[7], value.m[11]);
}

__host__ __device__ inline double2x4 transpose(const double4x2& value) {
  return double2x4(value.m[0], value.m[2], value.m[4], value.m[6], value.m[1],
                   value.m[3], value.m[5], value.m[7]);
}

__host__ __device__ inline double3x4 transpose(const double4x3& value) {
  return double3x4(value.m[0], value.m[3], value.m[6], value.m[9], value.m[1],
                   value.m[4], value.m[7], value.m[10], value.m[2], value.m[5],
                   value.m[8], value.m[11]);
}

__host__ __device__ inline double4x4 transpose(const double4x4& value) {
  return double4x4(value.m[0], value.m[4], value.m[8], value.m[12], value.m[1],
                   value.m[5], value.m[9], value.m[13], value.m[2], value.m[6],
                   value.m[10], value.m[14], value.m[3], value.m[7],
                   value.m[11], value.m[15]);
}

__host__ __device__ inline double2x2 inverse(const double2x2& value) {
  double2x2 a(value);
  double2x2 result(double(1));
  for (int column = 0; column < 2; ++column) {
    int pivot_row = column;
    double pivot_value = a.m[column * 2 + column];
    double pivot_abs = pivot_value < double(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 2; ++row) {
      double candidate_value = a.m[column * 2 + row];
      double candidate_abs =
          candidate_value < double(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == double(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 2; ++c) {
        double tmp = a.m[c * 2 + column];
        a.m[c * 2 + column] = a.m[c * 2 + pivot_row];
        a.m[c * 2 + pivot_row] = tmp;
        tmp = result.m[c * 2 + column];
        result.m[c * 2 + column] = result.m[c * 2 + pivot_row];
        result.m[c * 2 + pivot_row] = tmp;
      }
    }
    double pivot = a.m[column * 2 + column];
    for (int c = 0; c < 2; ++c) {
      a.m[c * 2 + column] /= pivot;
      result.m[c * 2 + column] /= pivot;
    }
    for (int row = 0; row < 2; ++row) {
      if (row == column) {
        continue;
      }
      double factor = a.m[column * 2 + row];
      for (int c = 0; c < 2; ++c) {
        a.m[c * 2 + row] -= factor * a.m[c * 2 + column];
        result.m[c * 2 + row] -= factor * result.m[c * 2 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline double3x3 inverse(const double3x3& value) {
  double3x3 a(value);
  double3x3 result(double(1));
  for (int column = 0; column < 3; ++column) {
    int pivot_row = column;
    double pivot_value = a.m[column * 3 + column];
    double pivot_abs = pivot_value < double(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 3; ++row) {
      double candidate_value = a.m[column * 3 + row];
      double candidate_abs =
          candidate_value < double(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == double(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 3; ++c) {
        double tmp = a.m[c * 3 + column];
        a.m[c * 3 + column] = a.m[c * 3 + pivot_row];
        a.m[c * 3 + pivot_row] = tmp;
        tmp = result.m[c * 3 + column];
        result.m[c * 3 + column] = result.m[c * 3 + pivot_row];
        result.m[c * 3 + pivot_row] = tmp;
      }
    }
    double pivot = a.m[column * 3 + column];
    for (int c = 0; c < 3; ++c) {
      a.m[c * 3 + column] /= pivot;
      result.m[c * 3 + column] /= pivot;
    }
    for (int row = 0; row < 3; ++row) {
      if (row == column) {
        continue;
      }
      double factor = a.m[column * 3 + row];
      for (int c = 0; c < 3; ++c) {
        a.m[c * 3 + row] -= factor * a.m[c * 3 + column];
        result.m[c * 3 + row] -= factor * result.m[c * 3 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline double4x4 inverse(const double4x4& value) {
  double4x4 a(value);
  double4x4 result(double(1));
  for (int column = 0; column < 4; ++column) {
    int pivot_row = column;
    double pivot_value = a.m[column * 4 + column];
    double pivot_abs = pivot_value < double(0) ? -pivot_value : pivot_value;
    for (int row = column + 1; row < 4; ++row) {
      double candidate_value = a.m[column * 4 + row];
      double candidate_abs =
          candidate_value < double(0) ? -candidate_value : candidate_value;
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (pivot_abs == double(0)) {
      return result;
    }
    if (pivot_row != column) {
      for (int c = 0; c < 4; ++c) {
        double tmp = a.m[c * 4 + column];
        a.m[c * 4 + column] = a.m[c * 4 + pivot_row];
        a.m[c * 4 + pivot_row] = tmp;
        tmp = result.m[c * 4 + column];
        result.m[c * 4 + column] = result.m[c * 4 + pivot_row];
        result.m[c * 4 + pivot_row] = tmp;
      }
    }
    double pivot = a.m[column * 4 + column];
    for (int c = 0; c < 4; ++c) {
      a.m[c * 4 + column] /= pivot;
      result.m[c * 4 + column] /= pivot;
    }
    for (int row = 0; row < 4; ++row) {
      if (row == column) {
        continue;
      }
      double factor = a.m[column * 4 + row];
      for (int c = 0; c < 4; ++c) {
        a.m[c * 4 + row] -= factor * a.m[c * 4 + column];
        result.m[c * 4 + row] -= factor * result.m[c * 4 + column];
      }
    }
  }
  return result;
}

__host__ __device__ inline double2 operator*(const double2x2& lhs,
                                             const double2& rhs) {
  return make_double2(lhs.m[0] * rhs.x + lhs.m[2] * rhs.y,
                      lhs.m[1] * rhs.x + lhs.m[3] * rhs.y);
}

__host__ __device__ inline double2 operator*(const double2& lhs,
                                             const double2x2& rhs) {
  return make_double2(lhs.x * rhs.m[0] + lhs.y * rhs.m[1],
                      lhs.x * rhs.m[2] + lhs.y * rhs.m[3]);
}

__host__ __device__ inline double2x2 operator*(const double2x2& lhs,
                                               const double2x2& rhs) {
  return double2x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3]);
}

__host__ __device__ inline double3x2 operator*(const double2x2& lhs,
                                               const double3x2& rhs) {
  return double3x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5]);
}

__host__ __device__ inline double4x2 operator*(const double2x2& lhs,
                                               const double4x2& rhs) {
  return double4x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[2] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                   lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7],
                   lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7]);
}

__host__ __device__ inline double3 operator*(const double2x3& lhs,
                                             const double2& rhs) {
  return make_double3(lhs.m[0] * rhs.x + lhs.m[3] * rhs.y,
                      lhs.m[1] * rhs.x + lhs.m[4] * rhs.y,
                      lhs.m[2] * rhs.x + lhs.m[5] * rhs.y);
}

__host__ __device__ inline double2 operator*(const double3& lhs,
                                             const double2x3& rhs) {
  return make_double2(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                      lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5]);
}

__host__ __device__ inline double2x3 operator*(const double2x3& lhs,
                                               const double2x2& rhs) {
  return double2x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3]);
}

__host__ __device__ inline double3x3 operator*(const double2x3& lhs,
                                               const double3x2& rhs) {
  return double3x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                   lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5]);
}

__host__ __device__ inline double4x3 operator*(const double2x3& lhs,
                                               const double4x2& rhs) {
  return double4x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[3] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                   lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                   lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7],
                   lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7],
                   lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7]);
}

__host__ __device__ inline double4 operator*(const double2x4& lhs,
                                             const double2& rhs) {
  return make_double4(
      lhs.m[0] * rhs.x + lhs.m[4] * rhs.y, lhs.m[1] * rhs.x + lhs.m[5] * rhs.y,
      lhs.m[2] * rhs.x + lhs.m[6] * rhs.y, lhs.m[3] * rhs.x + lhs.m[7] * rhs.y);
}

__host__ __device__ inline double2 operator*(const double4& lhs,
                                             const double2x4& rhs) {
  return make_double2(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] +
          lhs.w * rhs.m[7]);
}

__host__ __device__ inline double2x4 operator*(const double2x4& lhs,
                                               const double2x2& rhs) {
  return double2x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3]);
}

__host__ __device__ inline double3x4 operator*(const double2x4& lhs,
                                               const double3x2& rhs) {
  return double3x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                   lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5],
                   lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5]);
}

__host__ __device__ inline double4x4 operator*(const double2x4& lhs,
                                               const double4x2& rhs) {
  return double4x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1],
                   lhs.m[0] * rhs.m[2] + lhs.m[4] * rhs.m[3],
                   lhs.m[1] * rhs.m[2] + lhs.m[5] * rhs.m[3],
                   lhs.m[2] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[3] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5],
                   lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5],
                   lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5],
                   lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5],
                   lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7],
                   lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7],
                   lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                   lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7]);
}

__host__ __device__ inline double2 operator*(const double3x2& lhs,
                                             const double3& rhs) {
  return make_double2(lhs.m[0] * rhs.x + lhs.m[2] * rhs.y + lhs.m[4] * rhs.z,
                      lhs.m[1] * rhs.x + lhs.m[3] * rhs.y + lhs.m[5] * rhs.z);
}

__host__ __device__ inline double3 operator*(const double2& lhs,
                                             const double3x2& rhs) {
  return make_double3(lhs.x * rhs.m[0] + lhs.y * rhs.m[1],
                      lhs.x * rhs.m[2] + lhs.y * rhs.m[3],
                      lhs.x * rhs.m[4] + lhs.y * rhs.m[5]);
}

__host__ __device__ inline double2x2 operator*(const double3x2& lhs,
                                               const double2x3& rhs) {
  return double2x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5]);
}

__host__ __device__ inline double3x2 operator*(const double3x2& lhs,
                                               const double3x3& rhs) {
  return double3x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7] + lhs.m[4] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[5] * rhs.m[8]);
}

__host__ __device__ inline double4x2 operator*(const double3x2& lhs,
                                               const double4x3& rhs) {
  return double4x2(
      lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] + lhs.m[4] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[5] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[2] * rhs.m[4] + lhs.m[4] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[5] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[2] * rhs.m[7] + lhs.m[4] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[5] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[2] * rhs.m[10] + lhs.m[4] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[3] * rhs.m[10] + lhs.m[5] * rhs.m[11]);
}

__host__ __device__ inline double3 operator*(const double3x3& lhs,
                                             const double3& rhs) {
  return make_double3(lhs.m[0] * rhs.x + lhs.m[3] * rhs.y + lhs.m[6] * rhs.z,
                      lhs.m[1] * rhs.x + lhs.m[4] * rhs.y + lhs.m[7] * rhs.z,
                      lhs.m[2] * rhs.x + lhs.m[5] * rhs.y + lhs.m[8] * rhs.z);
}

__host__ __device__ inline double3 operator*(const double3& lhs,
                                             const double3x3& rhs) {
  return make_double3(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                      lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5],
                      lhs.x * rhs.m[6] + lhs.y * rhs.m[7] + lhs.z * rhs.m[8]);
}

__host__ __device__ inline double2x3 operator*(const double3x3& lhs,
                                               const double2x3& rhs) {
  return double2x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5]);
}

__host__ __device__ inline double3x3 operator*(const double3x3& lhs,
                                               const double3x3& rhs) {
  return double3x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[6] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[7] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[8] * rhs.m[8]);
}

__host__ __device__ inline double4x3 operator*(const double3x3& lhs,
                                               const double4x3& rhs) {
  return double4x3(
      lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] + lhs.m[6] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[7] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[3] * rhs.m[4] + lhs.m[6] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[7] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[3] * rhs.m[7] + lhs.m[6] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[7] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[3] * rhs.m[10] + lhs.m[6] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[4] * rhs.m[10] + lhs.m[7] * rhs.m[11],
      lhs.m[2] * rhs.m[9] + lhs.m[5] * rhs.m[10] + lhs.m[8] * rhs.m[11]);
}

__host__ __device__ inline double4 operator*(const double3x4& lhs,
                                             const double3& rhs) {
  return make_double4(lhs.m[0] * rhs.x + lhs.m[4] * rhs.y + lhs.m[8] * rhs.z,
                      lhs.m[1] * rhs.x + lhs.m[5] * rhs.y + lhs.m[9] * rhs.z,
                      lhs.m[2] * rhs.x + lhs.m[6] * rhs.y + lhs.m[10] * rhs.z,
                      lhs.m[3] * rhs.x + lhs.m[7] * rhs.y + lhs.m[11] * rhs.z);
}

__host__ __device__ inline double3 operator*(const double4& lhs,
                                             const double3x4& rhs) {
  return make_double3(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] + lhs.w * rhs.m[7],
      lhs.x * rhs.m[8] + lhs.y * rhs.m[9] + lhs.z * rhs.m[10] +
          lhs.w * rhs.m[11]);
}

__host__ __device__ inline double2x4 operator*(const double3x4& lhs,
                                               const double2x3& rhs) {
  return double2x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5]);
}

__host__ __device__ inline double3x4 operator*(const double3x4& lhs,
                                               const double3x3& rhs) {
  return double3x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[9] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7] + lhs.m[10] * rhs.m[8],
      lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7] + lhs.m[11] * rhs.m[8]);
}

__host__ __device__ inline double4x4 operator*(const double3x4& lhs,
                                               const double4x3& rhs) {
  return double4x4(
      lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] + lhs.m[8] * rhs.m[2],
      lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] + lhs.m[9] * rhs.m[2],
      lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] + lhs.m[10] * rhs.m[2],
      lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] + lhs.m[11] * rhs.m[2],
      lhs.m[0] * rhs.m[3] + lhs.m[4] * rhs.m[4] + lhs.m[8] * rhs.m[5],
      lhs.m[1] * rhs.m[3] + lhs.m[5] * rhs.m[4] + lhs.m[9] * rhs.m[5],
      lhs.m[2] * rhs.m[3] + lhs.m[6] * rhs.m[4] + lhs.m[10] * rhs.m[5],
      lhs.m[3] * rhs.m[3] + lhs.m[7] * rhs.m[4] + lhs.m[11] * rhs.m[5],
      lhs.m[0] * rhs.m[6] + lhs.m[4] * rhs.m[7] + lhs.m[8] * rhs.m[8],
      lhs.m[1] * rhs.m[6] + lhs.m[5] * rhs.m[7] + lhs.m[9] * rhs.m[8],
      lhs.m[2] * rhs.m[6] + lhs.m[6] * rhs.m[7] + lhs.m[10] * rhs.m[8],
      lhs.m[3] * rhs.m[6] + lhs.m[7] * rhs.m[7] + lhs.m[11] * rhs.m[8],
      lhs.m[0] * rhs.m[9] + lhs.m[4] * rhs.m[10] + lhs.m[8] * rhs.m[11],
      lhs.m[1] * rhs.m[9] + lhs.m[5] * rhs.m[10] + lhs.m[9] * rhs.m[11],
      lhs.m[2] * rhs.m[9] + lhs.m[6] * rhs.m[10] + lhs.m[10] * rhs.m[11],
      lhs.m[3] * rhs.m[9] + lhs.m[7] * rhs.m[10] + lhs.m[11] * rhs.m[11]);
}

__host__ __device__ inline double2 operator*(const double4x2& lhs,
                                             const double4& rhs) {
  return make_double2(
      lhs.m[0] * rhs.x + lhs.m[2] * rhs.y + lhs.m[4] * rhs.z + lhs.m[6] * rhs.w,
      lhs.m[1] * rhs.x + lhs.m[3] * rhs.y + lhs.m[5] * rhs.z +
          lhs.m[7] * rhs.w);
}

__host__ __device__ inline double4 operator*(const double2& lhs,
                                             const double4x2& rhs) {
  return make_double4(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1], lhs.x * rhs.m[2] + lhs.y * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5], lhs.x * rhs.m[6] + lhs.y * rhs.m[7]);
}

__host__ __device__ inline double2x2 operator*(const double4x2& lhs,
                                               const double2x4& rhs) {
  return double2x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                       lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                       lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7]);
}

__host__ __device__ inline double3x2 operator*(const double4x2& lhs,
                                               const double3x4& rhs) {
  return double3x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                       lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                       lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[2] * rhs.m[9] +
                       lhs.m[4] * rhs.m[10] + lhs.m[6] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                       lhs.m[5] * rhs.m[10] + lhs.m[7] * rhs.m[11]);
}

__host__ __device__ inline double4x2 operator*(const double4x2& lhs,
                                               const double4x4& rhs) {
  return double4x2(lhs.m[0] * rhs.m[0] + lhs.m[2] * rhs.m[1] +
                       lhs.m[4] * rhs.m[2] + lhs.m[6] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[5] * rhs.m[2] + lhs.m[7] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[2] * rhs.m[5] +
                       lhs.m[4] * rhs.m[6] + lhs.m[6] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[5] * rhs.m[6] + lhs.m[7] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[2] * rhs.m[9] +
                       lhs.m[4] * rhs.m[10] + lhs.m[6] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                       lhs.m[5] * rhs.m[10] + lhs.m[7] * rhs.m[11],
                   lhs.m[0] * rhs.m[12] + lhs.m[2] * rhs.m[13] +
                       lhs.m[4] * rhs.m[14] + lhs.m[6] * rhs.m[15],
                   lhs.m[1] * rhs.m[12] + lhs.m[3] * rhs.m[13] +
                       lhs.m[5] * rhs.m[14] + lhs.m[7] * rhs.m[15]);
}

__host__ __device__ inline double3 operator*(const double4x3& lhs,
                                             const double4& rhs) {
  return make_double3(
      lhs.m[0] * rhs.x + lhs.m[3] * rhs.y + lhs.m[6] * rhs.z + lhs.m[9] * rhs.w,
      lhs.m[1] * rhs.x + lhs.m[4] * rhs.y + lhs.m[7] * rhs.z +
          lhs.m[10] * rhs.w,
      lhs.m[2] * rhs.x + lhs.m[5] * rhs.y + lhs.m[8] * rhs.z +
          lhs.m[11] * rhs.w);
}

__host__ __device__ inline double4 operator*(const double3& lhs,
                                             const double4x3& rhs) {
  return make_double4(lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2],
                      lhs.x * rhs.m[3] + lhs.y * rhs.m[4] + lhs.z * rhs.m[5],
                      lhs.x * rhs.m[6] + lhs.y * rhs.m[7] + lhs.z * rhs.m[8],
                      lhs.x * rhs.m[9] + lhs.y * rhs.m[10] + lhs.z * rhs.m[11]);
}

__host__ __device__ inline double2x3 operator*(const double4x3& lhs,
                                               const double2x4& rhs) {
  return double2x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7]);
}

__host__ __device__ inline double3x3 operator*(const double4x3& lhs,
                                               const double3x4& rhs) {
  return double3x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                       lhs.m[6] * rhs.m[10] + lhs.m[9] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                       lhs.m[7] * rhs.m[10] + lhs.m[10] * rhs.m[11],
                   lhs.m[2] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                       lhs.m[8] * rhs.m[10] + lhs.m[11] * rhs.m[11]);
}

__host__ __device__ inline double4x3 operator*(const double4x3& lhs,
                                               const double4x4& rhs) {
  return double4x3(lhs.m[0] * rhs.m[0] + lhs.m[3] * rhs.m[1] +
                       lhs.m[6] * rhs.m[2] + lhs.m[9] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[7] * rhs.m[2] + lhs.m[10] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[11] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[3] * rhs.m[5] +
                       lhs.m[6] * rhs.m[6] + lhs.m[9] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[7] * rhs.m[6] + lhs.m[10] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[11] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[3] * rhs.m[9] +
                       lhs.m[6] * rhs.m[10] + lhs.m[9] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                       lhs.m[7] * rhs.m[10] + lhs.m[10] * rhs.m[11],
                   lhs.m[2] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                       lhs.m[8] * rhs.m[10] + lhs.m[11] * rhs.m[11],
                   lhs.m[0] * rhs.m[12] + lhs.m[3] * rhs.m[13] +
                       lhs.m[6] * rhs.m[14] + lhs.m[9] * rhs.m[15],
                   lhs.m[1] * rhs.m[12] + lhs.m[4] * rhs.m[13] +
                       lhs.m[7] * rhs.m[14] + lhs.m[10] * rhs.m[15],
                   lhs.m[2] * rhs.m[12] + lhs.m[5] * rhs.m[13] +
                       lhs.m[8] * rhs.m[14] + lhs.m[11] * rhs.m[15]);
}

__host__ __device__ inline double4 operator*(const double4x4& lhs,
                                             const double4& rhs) {
  return make_double4(lhs.m[0] * rhs.x + lhs.m[4] * rhs.y + lhs.m[8] * rhs.z +
                          lhs.m[12] * rhs.w,
                      lhs.m[1] * rhs.x + lhs.m[5] * rhs.y + lhs.m[9] * rhs.z +
                          lhs.m[13] * rhs.w,
                      lhs.m[2] * rhs.x + lhs.m[6] * rhs.y + lhs.m[10] * rhs.z +
                          lhs.m[14] * rhs.w,
                      lhs.m[3] * rhs.x + lhs.m[7] * rhs.y + lhs.m[11] * rhs.z +
                          lhs.m[15] * rhs.w);
}

__host__ __device__ inline double4 operator*(const double4& lhs,
                                             const double4x4& rhs) {
  return make_double4(
      lhs.x * rhs.m[0] + lhs.y * rhs.m[1] + lhs.z * rhs.m[2] + lhs.w * rhs.m[3],
      lhs.x * rhs.m[4] + lhs.y * rhs.m[5] + lhs.z * rhs.m[6] + lhs.w * rhs.m[7],
      lhs.x * rhs.m[8] + lhs.y * rhs.m[9] + lhs.z * rhs.m[10] +
          lhs.w * rhs.m[11],
      lhs.x * rhs.m[12] + lhs.y * rhs.m[13] + lhs.z * rhs.m[14] +
          lhs.w * rhs.m[15]);
}

__host__ __device__ inline double2x4 operator*(const double4x4& lhs,
                                               const double2x4& rhs) {
  return double2x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                       lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                       lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                       lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                   lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                       lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7]);
}

__host__ __device__ inline double3x4 operator*(const double4x4& lhs,
                                               const double3x4& rhs) {
  return double3x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                       lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                       lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                       lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                   lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                       lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                       lhs.m[8] * rhs.m[10] + lhs.m[12] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                       lhs.m[9] * rhs.m[10] + lhs.m[13] * rhs.m[11],
                   lhs.m[2] * rhs.m[8] + lhs.m[6] * rhs.m[9] +
                       lhs.m[10] * rhs.m[10] + lhs.m[14] * rhs.m[11],
                   lhs.m[3] * rhs.m[8] + lhs.m[7] * rhs.m[9] +
                       lhs.m[11] * rhs.m[10] + lhs.m[15] * rhs.m[11]);
}

__host__ __device__ inline double4x4 operator*(const double4x4& lhs,
                                               const double4x4& rhs) {
  return double4x4(lhs.m[0] * rhs.m[0] + lhs.m[4] * rhs.m[1] +
                       lhs.m[8] * rhs.m[2] + lhs.m[12] * rhs.m[3],
                   lhs.m[1] * rhs.m[0] + lhs.m[5] * rhs.m[1] +
                       lhs.m[9] * rhs.m[2] + lhs.m[13] * rhs.m[3],
                   lhs.m[2] * rhs.m[0] + lhs.m[6] * rhs.m[1] +
                       lhs.m[10] * rhs.m[2] + lhs.m[14] * rhs.m[3],
                   lhs.m[3] * rhs.m[0] + lhs.m[7] * rhs.m[1] +
                       lhs.m[11] * rhs.m[2] + lhs.m[15] * rhs.m[3],
                   lhs.m[0] * rhs.m[4] + lhs.m[4] * rhs.m[5] +
                       lhs.m[8] * rhs.m[6] + lhs.m[12] * rhs.m[7],
                   lhs.m[1] * rhs.m[4] + lhs.m[5] * rhs.m[5] +
                       lhs.m[9] * rhs.m[6] + lhs.m[13] * rhs.m[7],
                   lhs.m[2] * rhs.m[4] + lhs.m[6] * rhs.m[5] +
                       lhs.m[10] * rhs.m[6] + lhs.m[14] * rhs.m[7],
                   lhs.m[3] * rhs.m[4] + lhs.m[7] * rhs.m[5] +
                       lhs.m[11] * rhs.m[6] + lhs.m[15] * rhs.m[7],
                   lhs.m[0] * rhs.m[8] + lhs.m[4] * rhs.m[9] +
                       lhs.m[8] * rhs.m[10] + lhs.m[12] * rhs.m[11],
                   lhs.m[1] * rhs.m[8] + lhs.m[5] * rhs.m[9] +
                       lhs.m[9] * rhs.m[10] + lhs.m[13] * rhs.m[11],
                   lhs.m[2] * rhs.m[8] + lhs.m[6] * rhs.m[9] +
                       lhs.m[10] * rhs.m[10] + lhs.m[14] * rhs.m[11],
                   lhs.m[3] * rhs.m[8] + lhs.m[7] * rhs.m[9] +
                       lhs.m[11] * rhs.m[10] + lhs.m[15] * rhs.m[11],
                   lhs.m[0] * rhs.m[12] + lhs.m[4] * rhs.m[13] +
                       lhs.m[8] * rhs.m[14] + lhs.m[12] * rhs.m[15],
                   lhs.m[1] * rhs.m[12] + lhs.m[5] * rhs.m[13] +
                       lhs.m[9] * rhs.m[14] + lhs.m[13] * rhs.m[15],
                   lhs.m[2] * rhs.m[12] + lhs.m[6] * rhs.m[13] +
                       lhs.m[10] * rhs.m[14] + lhs.m[14] * rhs.m[15],
                   lhs.m[3] * rhs.m[12] + lhs.m[7] * rhs.m[13] +
                       lhs.m[11] * rhs.m[14] + lhs.m[15] * rhs.m[15]);
}

struct Matrix32 {
  float data[32][32];
};

struct Matrix64 {
  float data[64][64];
};

struct Matrix128 {
  float data[128][128];
};

struct MatrixDimensions {
  int rows_A;
  int cols_A;
  int rows_B;
  int cols_B;
  int rows_C;
  int cols_C;
};

struct KernelParams {
  int block_size;
  int grid_size_x;
  int grid_size_y;
  float alpha;
  float beta;
};
const int TILE_SIZE = 16;
const int MAX_SHARED_MEMORY = 1024;
const int WARP_SIZE = 32;
__constant__ MatrixDimensions dims;
__constant__ KernelParams params;
float* matrix_A;
float* matrix_B;
float* matrix_C;
__constant__ int matrix_size;
__constant__ float alpha;
__constant__ float beta;
float* A;
float* B;
float* C;
float4* matrix_A_vec;
float4* matrix_B_vec;
float4* matrix_C_vec;
__constant__ int batch_size;
float* batch_A;
float* batch_B;
float* batch_C;
__device__ float getMatrixElement(float* matrix, int row, int col, int stride) {
  return matrix[((row * stride) + col)];
}

__device__ void setMatrixElement(float* matrix, int row, int col, int stride,
                                 float value) {
  matrix[((row * stride) + col)] = value;
}

__global__ void matmul_tiled() {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
  int tx = int(threadIdx.x);
  int ty = int(threadIdx.y);
  int bx = int(blockIdx.x);
  int by = int(blockIdx.y);
  int row = ((by * TILE_SIZE) + ty);
  int col = ((bx * TILE_SIZE) + tx);
  float result = 0.0;
  int num_tiles = (((dims.cols_A + TILE_SIZE) - 1) / TILE_SIZE);
  for (int tile = 0; (tile < num_tiles); tile++) {
    int a_row = row;
    int a_col = ((tile * TILE_SIZE) + tx);
    if (((a_row < dims.rows_A) && (a_col < dims.cols_A))) {
      tile_A[ty][tx] = getMatrixElement(matrix_A, a_row, a_col, dims.cols_A);
    } else {
      tile_A[ty][tx] = 0.0;
    }
    int b_row = ((tile * TILE_SIZE) + ty);
    int b_col = col;
    if (((b_row < dims.rows_B) && (b_col < dims.cols_B))) {
      tile_B[ty][tx] = getMatrixElement(matrix_B, b_row, b_col, dims.cols_B);
    } else {
      tile_B[ty][tx] = 0.0;
    }
    __syncthreads();
    for (int k = 0; (k < TILE_SIZE); k++) {
      result += (tile_A[ty][k] * tile_B[k][tx]);
    }
    __syncthreads();
  }
  if (((row < dims.rows_C) && (col < dims.cols_C))) {
    float existing_value = getMatrixElement(matrix_C, row, col, dims.cols_C);
    float final_result =
        ((params.alpha * result) + (params.beta * existing_value));
    setMatrixElement(matrix_C, row, col, dims.cols_C, final_result);
  }
}

__global__ void matmul_square() {
  __shared__ float shared_A[16][16];
  __shared__ float shared_B[16][16];
  int tx = int(threadIdx.x);
  int ty = int(threadIdx.y);
  int bx = int(blockIdx.x);
  int by = int(blockIdx.y);
  int row = ((by * 16) + ty);
  int col = ((bx * 16) + tx);
  float sum = 0.0;
  for (int tile = 0; (tile < ((matrix_size + 15) / 16)); tile++) {
    int a_col = ((tile * 16) + tx);
    int b_row = ((tile * 16) + ty);
    shared_A[ty][tx] = (((row < matrix_size) && (a_col < matrix_size))
                            ? A[((row * matrix_size) + a_col)]
                            : 0.0);
    shared_B[ty][tx] = (((b_row < matrix_size) && (col < matrix_size))
                            ? B[((b_row * matrix_size) + col)]
                            : 0.0);
    __syncthreads();
    for (int k = 0; (k < 16); k++) {
      sum += (shared_A[ty][k] * shared_B[k][tx]);
    }
    __syncthreads();
  }
  if (((row < matrix_size) && (col < matrix_size))) {
    C[((row * matrix_size) + col)] =
        ((alpha * sum) + (beta * C[((row * matrix_size) + col)]));
  }
}

__global__ void matmul_vectorized() {
  __shared__ float4 shared_A_vec[8][8];
  __shared__ float4 shared_B_vec[8][8];
  int tx = int(threadIdx.x);
  int ty = int(threadIdx.y);
  int bx = int(blockIdx.x);
  int by = int(blockIdx.y);
  int row = ((by * 8) + ty);
  int col = ((bx * 8) + tx);
  float4 result = make_float4(0.0, 0.0, 0.0, 0.0);
  for (int tile = 0; (tile < ((dims.cols_A + 31) / 32)); tile++) {
    int a_col_vec = ((tile * 8) + tx);
    int b_row_vec = ((tile * 8) + ty);
    if (((row < (dims.rows_A / 4)) && (a_col_vec < (dims.cols_A / 4)))) {
      shared_A_vec[ty][tx] =
          matrix_A_vec[((row * (dims.cols_A / 4)) + a_col_vec)];
    } else {
      shared_A_vec[ty][tx] = make_float4(0.0, 0.0, 0.0, 0.0);
    }
    if (((b_row_vec < (dims.rows_B / 4)) && (col < (dims.cols_B / 4)))) {
      shared_B_vec[ty][tx] =
          matrix_B_vec[((b_row_vec * (dims.cols_B / 4)) + col)];
    } else {
      shared_B_vec[ty][tx] = make_float4(0.0, 0.0, 0.0, 0.0);
    }
    __syncthreads();
    for (int k = 0; (k < 8); k++) {
      result = cgl_float4_add(
          result, cgl_float4_mul(shared_A_vec[ty][k], shared_B_vec[k][tx]));
    }
    __syncthreads();
  }
  if (((row < (dims.rows_C / 4)) && (col < (dims.cols_C / 4)))) {
    float4 existing = matrix_C_vec[((row * (dims.cols_C / 4)) + col)];
    matrix_C_vec[((row * (dims.cols_C / 4)) + col)] =
        cgl_float4_add(cgl_scalar_mul_float4(params.alpha, result),
                       cgl_scalar_mul_float4(params.beta, existing));
  }
}

__global__ void matmul_batched() {
  __shared__ float tile_A[16][16];
  __shared__ float tile_B[16][16];
  int tx = int(threadIdx.x);
  int ty = int(threadIdx.y);
  int bx = int(blockIdx.x);
  int by = int(blockIdx.y);
  int bz = int(blockIdx.z);
  int batch_id = bz;
  if ((batch_id >= batch_size)) {
    return;
  }
  int matrix_offset = ((batch_id * matrix_size) * matrix_size);
  int row = ((by * 16) + ty);
  int col = ((bx * 16) + tx);
  float result = 0.0;
  for (int tile = 0; (tile < ((matrix_size + 15) / 16)); tile++) {
    int a_col = ((tile * 16) + tx);
    int b_row = ((tile * 16) + ty);
    if (((row < matrix_size) && (a_col < matrix_size))) {
      tile_A[ty][tx] = batch_A[((matrix_offset + (row * matrix_size)) + a_col)];
    } else {
      tile_A[ty][tx] = 0.0;
    }
    if (((b_row < matrix_size) && (col < matrix_size))) {
      tile_B[ty][tx] = batch_B[((matrix_offset + (b_row * matrix_size)) + col)];
    } else {
      tile_B[ty][tx] = 0.0;
    }
    __syncthreads();
    for (int k = 0; (k < 16); k++) {
      result += (tile_A[ty][k] * tile_B[k][tx]);
    }
    __syncthreads();
  }
  if (((row < matrix_size) && (col < matrix_size))) {
    int c_index = ((matrix_offset + (row * matrix_size)) + col);
    batch_C[c_index] = ((alpha * result) + (beta * batch_C[c_index]));
  }
}

__global__ void matmul_warp_optimized() {
  int warp_id = int(threadIdx.x);
  int global_id = int((blockIdx.x * blockDim.x + threadIdx.x));
  int elements_per_warp = ((matrix_size * matrix_size) / (int(gridDim.x) * 32));
  for (int elem = 0; (elem < elements_per_warp); elem++) {
    int linear_id = (global_id + ((elem * int(gridDim.x)) * 32));
    if ((linear_id >= (matrix_size * matrix_size))) {
      break;
    }
    int row = (linear_id / matrix_size);
    int col = (linear_id % matrix_size);
    float result = 0.0;
    for (int k = 0; (k < matrix_size); k += WARP_SIZE) {
      float a_val = (((k + warp_id) < matrix_size)
                         ? A[(((row * matrix_size) + k) + warp_id)]
                         : 0.0);
      float b_val = (((k + warp_id) < matrix_size)
                         ? B[(((k + warp_id) * matrix_size) + col)]
                         : 0.0);
      float partial_sum = (a_val * b_val);
      for (int offset = 16; (offset > 0); offset /= 2) {
        partial_sum +=
            cgl_hip_shfl_down_sync_float(4294967295, partial_sum, offset);
      }
      if ((warp_id == 0)) {
        result += partial_sum;
      }
    }
    if ((warp_id == 0)) {
      C[linear_id] = ((alpha * result) + (beta * C[linear_id]));
    }
  }
}
