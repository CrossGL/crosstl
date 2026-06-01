#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

__device__ inline float cgl_float3_dot(float3 lhs, float3 rhs) {
  return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

__device__ inline float3 cgl_scalar_sub_float3(float lhs, float3 rhs) {
  return make_float3((lhs - rhs.x), (lhs - rhs.y), (lhs - rhs.z));
}

__device__ inline float3 cgl_float3_mul_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x * rhs), (lhs.y * rhs), (lhs.z * rhs));
}

__device__ inline float3 cgl_float3_add(float3 lhs, float3 rhs) {
  return make_float3((lhs.x + rhs.x), (lhs.y + rhs.y), (lhs.z + rhs.z));
}

__device__ inline float3 cgl_float3_floor(float3 value) {
  return make_float3(floorf(value.x), floorf(value.y), floorf(value.z));
}

__device__ inline float cgl_fract_float(float value) {
  return value - floorf(value);
}

__device__ inline float3 cgl_float3_fract(float3 value) {
  return make_float3(cgl_fract_float(value.x), cgl_fract_float(value.y),
                     cgl_fract_float(value.z));
}

__device__ inline float3 cgl_float3_mul(float3 lhs, float3 rhs) {
  return make_float3((lhs.x * rhs.x), (lhs.y * rhs.y), (lhs.z * rhs.z));
}

__device__ inline float3 cgl_float3_sub_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x - rhs), (lhs.y - rhs), (lhs.z - rhs));
}

__device__ inline float3 cgl_float3_add_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x + rhs), (lhs.y + rhs), (lhs.z + rhs));
}

__device__ inline float3 cgl_float3_abs(float3 value) {
  return make_float3(fabsf(value.x), fabsf(value.y), fabsf(value.z));
}

__device__ inline float2 cgl_float2_mul_scalar(float2 lhs, float rhs) {
  return make_float2((lhs.x * rhs), (lhs.y * rhs));
}

__device__ inline float2 cgl_float2_add_scalar(float2 lhs, float rhs) {
  return make_float2((lhs.x + rhs), (lhs.y + rhs));
}

__device__ inline float cgl_float3_length(float3 value) {
  return sqrtf((value.x * value.x) + (value.y * value.y) + (value.z * value.z));
}

__device__ inline float3 cgl_float3_normalize(float3 value) {
  float inv_length = 1.0f / cgl_float3_length(value);
  return make_float3((value.x * inv_length), (value.y * inv_length),
                     (value.z * inv_length));
}

__device__ inline float3 cgl_float3_sub(float3 lhs, float3 rhs) {
  return make_float3((lhs.x - rhs.x), (lhs.y - rhs.y), (lhs.z - rhs.z));
}

__device__ inline float4 cgl_float4_mul(float4 lhs, float4 rhs) {
  return make_float4((lhs.x * rhs.x), (lhs.y * rhs.y), (lhs.z * rhs.z),
                     (lhs.w * rhs.w));
}

__device__ inline float3 cgl_float3_div_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x / rhs), (lhs.y / rhs), (lhs.z / rhs));
}

__device__ inline float2 cgl_scalar_div_float2(float lhs, float2 rhs) {
  return make_float2((lhs / rhs.x), (lhs / rhs.y));
}

__device__ inline float2 cgl_float2_mul(float2 lhs, float2 rhs) {
  return make_float2((lhs.x * rhs.x), (lhs.y * rhs.y));
}

__device__ inline float2 cgl_float2_add(float2 lhs, float2 rhs) {
  return make_float2((lhs.x + rhs.x), (lhs.y + rhs.y));
}

__device__ inline float3 cgl_float3_mix_scalar(float3 x, float3 y, float a) {
  return make_float3((x.x + ((y.x - x.x) * a)), (x.y + ((y.y - x.y) * a)),
                     (x.z + ((y.z - x.z) * a)));
}

__device__ inline float3 cgl_scalar_mul_float3(float lhs, float3 rhs) {
  return make_float3((lhs * rhs.x), (lhs * rhs.y), (lhs * rhs.z));
}

__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs) {
  return make_float4((lhs.x + rhs.x), (lhs.y + rhs.y), (lhs.z + rhs.z),
                     (lhs.w + rhs.w));
}

__device__ inline float3 cgl_float3_div(float3 lhs, float3 rhs) {
  return make_float3((lhs.x / rhs.x), (lhs.y / rhs.y), (lhs.z / rhs.z));
}

__device__ inline float3 cgl_float3_splat(float value) {
  return make_float3(value, value, value);
}

__device__ inline float3 cgl_float3_pow_vector(float3 base, float3 exponent) {
  return make_float3(powf(base.x, exponent.x), powf(base.y, exponent.y),
                     powf(base.z, exponent.z));
}

__device__ inline float4 cgl_float4_construct_float3_xyz_float(float3 arg0,
                                                               float arg1) {
  return make_float4(arg0.x, arg0.y, arg0.z, arg1);
}

__device__ inline float2 cgl_float2_div(float2 lhs, float2 rhs) {
  return make_float2((lhs.x / rhs.x), (lhs.y / rhs.y));
}

__device__ inline float2 cgl_float2_sub(float2 lhs, float2 rhs) {
  return make_float2((lhs.x - rhs.x), (lhs.y - rhs.y));
}

__device__ inline float cgl_float2_length(float2 value) {
  return sqrtf((value.x * value.x) + (value.y * value.y));
}

__device__ inline float2 cgl_float2_normalize(float2 value) {
  float inv_length = 1.0f / cgl_float2_length(value);
  return make_float2((value.x * inv_length), (value.y * inv_length));
}

__device__ inline float3 cgl_float3_construct_float2_xy_float(float2 arg0,
                                                              float arg1) {
  return make_float3(arg0.x, arg0.y, arg1);
}

__device__ inline float2 cgl_float2_sub_scalar(float2 lhs, float rhs) {
  return make_float2((lhs.x - rhs), (lhs.y - rhs));
}

__device__ inline float cgl_smoothstep_float(float edge0, float edge1,
                                             float value) {
  float t = fmaxf(0.0f, fminf(1.0f, (value - edge0) / (edge1 - edge0)));
  return t * t * (3.0f - 2.0f * t);
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

struct Material {
  float3 albedo;
  float roughness;
  float metallic;
  float3 emissive;
  float opacity;
  bool hasNormalMap;
  hipTextureObject_t albedoMap;
  hipTextureObject_t normalMap;
  hipTextureObject_t metallicRoughnessMap;
};

struct Light {
  float3 position;
  float3 color;
  float intensity;
  float radius;
  bool castShadows;
  float4x4 viewProjection;
};

struct Scene {
  Material materials[4];
  Light lights[8];
  float3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  float4x4 viewMatrix;
  float4x4 projectionMatrix;
};

struct VertexInput {
  float3 position;
  float3 normal;
  float3 tangent;
  float3 bitangent;
  float2 texCoord0;
  float2 texCoord1;
  float4 color;
  int materialIndex;
};

struct VertexOutput {
  float3 worldPosition;
  float3 worldNormal;
  float3 worldTangent;
  float3 worldBitangent;
  float2 texCoord0;
  float2 texCoord1;
  float4 color;
  float3x3 TBN;
  int materialIndex;
  float4 clipPosition;
};

struct FragmentOutput {
  float4 color;
  float4 normalBuffer;
  float4 positionBuffer;
  float depth;
};

struct GlobalUniforms {
  Scene scene;
  float3 cameraPosition;
  float globalRoughness;
  float2 screenSize;
  float nearPlane;
  float farPlane;
  int frameCount;
  float* noiseValues;
};
const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const int MAX_ITERATIONS = 64;
const float3 UP_VECTOR = make_float3(0.0, 1.0, 0.0);
__constant__ GlobalUniforms globals;
// CrossGL resource metadata: name=shadowMap kind=texture set=0 binding=0
// binding_source=automatic
__constant__ hipTextureObject_t shadowMap;
// CrossGL resource metadata: name=outputImage kind=image set=0 binding=0
// binding_source=automatic
hipSurfaceObject_t outputImage;
__device__ float distributionGGX(float3 N, float3 H, float roughness) {
  float a = (roughness * roughness);
  float a2 = (a * a);
  float NdotH = fmaxf(cgl_float3_dot(N, H), 0.0);
  float NdotH2 = (NdotH * NdotH);
  float num = a2;
  float denom = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / fmaxf(denom, EPSILON));
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = ((r * r) / 8.0);
  float num = NdotV;
  float denom = ((NdotV * (1.0 - k)) + k);
  return (num / fmaxf(denom, EPSILON));
}

__device__ float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
  float NdotV = fmaxf(cgl_float3_dot(N, V), 0.0);
  float NdotL = fmaxf(cgl_float3_dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

__device__ float3 fresnelSchlick(float cosTheta, float3 F0) {
  return cgl_float3_add(
      F0, cgl_float3_mul_scalar(cgl_scalar_sub_float3(1.0, F0),
                                powf(fmaxf((1.0 - cosTheta), 0.0), 5.0)));
}

__device__ float noise3D(float3 p) {
  float3 i = cgl_float3_floor(p);
  float3 f = cgl_float3_fract(p);
  float3 u = cgl_float3_mul(
      cgl_float3_mul(cgl_float3_mul(f, f), f),
      cgl_float3_add_scalar(
          cgl_float3_mul(
              f, cgl_float3_sub_scalar(cgl_float3_mul_scalar(f, 6.0), 15.0)),
          10.0));
  float n000 = cgl_fract_float(
      (sinf(cgl_float3_dot(i, make_float3(13.534, 43.5234, 243.32))) * 4453.0));
  float n001 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(0.0, 0.0, 1.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n010 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(0.0, 1.0, 0.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n011 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(0.0, 1.0, 1.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n100 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(1.0, 0.0, 0.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n101 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(1.0, 0.0, 1.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n110 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(1.0, 1.0, 0.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n111 = cgl_fract_float(
      (sinf(cgl_float3_dot(cgl_float3_add(i, make_float3(1.0, 1.0, 1.0)),
                           make_float3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n00 = (n000 + ((n001 - n000) * u.z));
  float n01 = (n010 + ((n011 - n010) * u.z));
  float n10 = (n100 + ((n101 - n100) * u.z));
  float n11 = (n110 + ((n111 - n110) * u.z));
  float n0 = (n00 + ((n01 - n00) * u.y));
  float n1 = (n10 + ((n11 - n10) * u.y));
  return (n0 + ((n1 - n0) * u.x));
}

__device__ float fbm(float3 p, int octaves, float lacunarity, float gain) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  for (int i = 0; (i < octaves); i++) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    sum += (amplitude * noise3D(cgl_float3_mul_scalar(p, frequency)));
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return sum;
}

__device__ float4 samplePlanarProjection(hipTextureObject_t tex,
                                         float3 worldPos, float3 normal) {
  float3 absNormal = cgl_float3_abs(normal);
  bool useX = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  bool useY = (!useX && (absNormal.y >= absNormal.z));
  float2 uv;
  if (useX) {
    uv = cgl_float2_add_scalar(
        cgl_float2_mul_scalar(make_float2(worldPos.z, worldPos.y), 0.5), 0.5);
    if ((normal.x < 0.0)) {
      uv.x = (1.0 - uv.x);
    }
  } else {
    if (useY) {
      uv = cgl_float2_add_scalar(
          cgl_float2_mul_scalar(make_float2(worldPos.x, worldPos.z), 0.5), 0.5);
      if ((normal.y < 0.0)) {
        uv.y = (1.0 - uv.y);
      }
    } else {
      uv = cgl_float2_add_scalar(
          cgl_float2_mul_scalar(make_float2(worldPos.x, worldPos.y), 0.5), 0.5);
      if ((normal.z < 0.0)) {
        uv.x = (1.0 - uv.x);
      }
    }
  }
  return tex2D<float4>(tex, uv.x, uv.y);
}

__device__ VertexOutput vertex_main(VertexInput input) {
  VertexOutput output;
  float4x4 modelMatrix = float4x4(1.0);
  float4x4 viewMatrix = globals.scene.viewMatrix;
  float4x4 projectionMatrix = globals.scene.projectionMatrix;
  float4x4 modelViewMatrix = (viewMatrix * modelMatrix);
  float4x4 modelViewProjectionMatrix = (projectionMatrix * modelViewMatrix);
  float3x3 normalMatrix = float3x3(transpose(inverse(modelMatrix)));
  float4 worldPosition =
      (modelMatrix *
       make_float4(input.position.x, input.position.y, input.position.z, 1.0));
  float3 worldNormal = cgl_float3_normalize((normalMatrix * input.normal));
  float3 worldTangent = cgl_float3_normalize((normalMatrix * input.tangent));
  float3 worldBitangent =
      cgl_float3_normalize((normalMatrix * input.bitangent));
  float3x3 TBN = float3x3(worldTangent, worldBitangent, worldNormal);
  float displacement =
      (fbm(cgl_float3_add_scalar(
               make_float3(worldPosition.x, worldPosition.y, worldPosition.z),
               (globals.scene.time * 0.1)),
           4, 2.0, 0.5) *
       0.1);
  if ((input.materialIndex > 0)) {
    float3 __crossgl_swizzle_value_0 =
        cgl_float3_mul_scalar(worldNormal, displacement);
    worldPosition.x = (worldPosition.x + __crossgl_swizzle_value_0.x);
    worldPosition.y = (worldPosition.y + __crossgl_swizzle_value_0.y);
    worldPosition.z = (worldPosition.z + __crossgl_swizzle_value_0.z);
  }
  float3 viewDir = cgl_float3_normalize(cgl_float3_sub(
      globals.cameraPosition,
      make_float3(worldPosition.x, worldPosition.y, worldPosition.z)));
  float fresnel =
      powf((1.0 - fmaxf(0.0, cgl_float3_dot(worldNormal, viewDir))), 5.0);
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color = cgl_float4_mul(input.color, make_float4(1.0, 1.0, 1.0, 1.0));
    for (int i = 0; (i < 4); i++) {
      if ((i >= (globals.frameCount % 5))) {
        break;
      }
      Light light = globals.scene.lights[i];
      float3 lightDir = cgl_float3_normalize(cgl_float3_sub(
          light.position,
          make_float3(worldPosition.x, worldPosition.y, worldPosition.z)));
      float lightDistance = cgl_float3_length(cgl_float3_sub(
          light.position,
          make_float3(worldPosition.x, worldPosition.y, worldPosition.z)));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      float3 __crossgl_swizzle_value_1 = cgl_float3_mul_scalar(
          cgl_float3_mul_scalar(
              cgl_float3_mul_scalar(light.color, lightIntensity),
              fmaxf(0.0, cgl_float3_dot(worldNormal, lightDir))),
          0.025);
      output.color.x = (output.color.x + __crossgl_swizzle_value_1.x);
      output.color.y = (output.color.y + __crossgl_swizzle_value_1.y);
      output.color.z = (output.color.z + __crossgl_swizzle_value_1.z);
    }
  } else {
    output.color = input.color;
    if ((globals.globalRoughness > 0.5)) {
      if ((fresnel > 0.7)) {
        output.color.w *= 0.8;
      } else {
        output.color.w *= 0.9;
      }
    }
  }
  output.worldPosition =
      make_float3(worldPosition.x, worldPosition.y, worldPosition.z);
  output.worldNormal = worldNormal;
  output.worldTangent = worldTangent;
  output.worldBitangent = worldBitangent;
  output.texCoord0 = input.texCoord0;
  output.texCoord1 = input.texCoord1;
  output.TBN = TBN;
  output.materialIndex = input.materialIndex;
  output.clipPosition =
      (modelViewProjectionMatrix *
       make_float4(input.position.x, input.position.y, input.position.z, 1.0));
  return output;
}

__device__ float shadowCalculation(float4 fragPosLightSpace, int iteration,
                                   VertexOutput input);

__device__ float shadowCalculation(float4 fragPosLightSpace, int iteration,
                                   VertexOutput input) {
  if ((iteration > 3)) {
    return 0.0;
  }
  float3 projCoords = cgl_float3_div_scalar(
      make_float3(fragPosLightSpace.x, fragPosLightSpace.y,
                  fragPosLightSpace.z),
      fragPosLightSpace.w);
  projCoords =
      cgl_float3_add_scalar(cgl_float3_mul_scalar(projCoords, 0.5), 0.5);
  float closestDepth =
      tex2D<float4>(shadowMap, make_float2(projCoords.x, projCoords.y).x,
                    make_float2(projCoords.x, projCoords.y).y)
          .x;
  float currentDepth = projCoords.z;
  float bias = fmaxf(
      (0.05 * (1.0 - cgl_float3_dot(
                         input.worldNormal,
                         cgl_float3_normalize(cgl_float3_sub(
                             globals.cameraPosition, input.worldPosition))))),
      0.005);
  float shadow = (((currentDepth - bias) > closestDepth) ? 1.0 : 0.0);
  float pcfDepth = 0.0;
  float2 texelSize = cgl_scalar_div_float2(
      1.0, make_float2(globals.screenSize.x, globals.screenSize.y));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  for (int x = -1; (x <= 1); x++) {
    for (int y = -1; (y <= 1); y++) {
      float pcfDepth =
          tex2D<float4>(
              shadowMap,
              cgl_float2_add(
                  cgl_float2_add(make_float2(projCoords.x, projCoords.y),
                                 cgl_float2_mul(make_float2(x, y), texelSize)),
                  make_float2(offset, offset))
                  .x,
              cgl_float2_add(
                  cgl_float2_add(make_float2(projCoords.x, projCoords.y),
                                 cgl_float2_mul(make_float2(x, y), texelSize)),
                  make_float2(offset, offset))
                  .y)
              .x;
      shadow += (((currentDepth - bias) > pcfDepth) ? 1.0 : 0.0);
    }
  }
  shadow /= 9.0;
  if ((projCoords.z > 1.0)) {
    shadow = 0.0;
  }
  return shadow;
}

__device__ FragmentOutput fragment_main(VertexOutput input) {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  float4 albedoValue =
      tex2D<float4>(material.albedoMap, input.texCoord0.x, input.texCoord0.y);
  float4 normalValue =
      tex2D<float4>(material.normalMap, input.texCoord0.x, input.texCoord0.y);
  float4 metallicRoughnessValue = tex2D<float4>(
      material.metallicRoughnessMap, input.texCoord0.x, input.texCoord0.y);
  float3 normal = cgl_float3_sub_scalar(
      cgl_float3_mul_scalar(
          make_float3(normalValue.x, normalValue.y, normalValue.z), 2.0),
      1.0);
  float3 worldNormal = cgl_float3_normalize((input.TBN * normal));
  float3 albedo =
      cgl_float3_mul(make_float3(albedoValue.x, albedoValue.y, albedoValue.z),
                     material.albedo);
  float metallic = (metallicRoughnessValue.z * material.metallic);
  float roughness = (metallicRoughnessValue.y * material.roughness);
  float ao = metallicRoughnessValue.x;
  float3 viewDir = cgl_float3_normalize(
      cgl_float3_sub(globals.cameraPosition, input.worldPosition));
  float3 F0 =
      cgl_float3_mix_scalar(make_float3(0.04, 0.04, 0.04), albedo, metallic);
  float3 Lo = make_float3(0.0, 0.0, 0.0);
  for (int i = 0; (i < globals.scene.activeLightCount); i++) {
    if ((i >= 8)) {
      break;
    }
    Light light = globals.scene.lights[i];
    float3 lightDir = cgl_float3_normalize(
        cgl_float3_sub(light.position, input.worldPosition));
    float3 halfway = cgl_float3_normalize(cgl_float3_add(viewDir, lightDir));
    float distance =
        cgl_float3_length(cgl_float3_sub(light.position, input.worldPosition));
    float attenuation = (1.0 / (distance * distance));
    float3 radiance = cgl_float3_mul_scalar(
        cgl_float3_mul_scalar(light.color, light.intensity), attenuation);
    float NDF = distributionGGX(worldNormal, halfway, roughness);
    float G = geometrySmith(worldNormal, viewDir, lightDir, roughness);
    float3 F = fresnelSchlick(fmaxf(cgl_float3_dot(halfway, viewDir), 0.0), F0);
    float3 kS = F;
    float3 kD = cgl_float3_sub(make_float3(1.0, 1.0, 1.0), kS);
    kD = cgl_float3_mul_scalar(kD, (1.0 - metallic));
    float3 numerator = cgl_scalar_mul_float3((NDF * G), F);
    float denominator =
        (((4.0 * fmaxf(cgl_float3_dot(worldNormal, viewDir), 0.0)) *
          fmaxf(cgl_float3_dot(worldNormal, lightDir), 0.0)) +
         EPSILON);
    float3 specular = cgl_float3_div_scalar(numerator, denominator);
    float NdotL = fmaxf(cgl_float3_dot(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      float4 fragPosLightSpace =
          (light.viewProjection * make_float4(input.worldPosition.x,
                                              input.worldPosition.y,
                                              input.worldPosition.z, 1.0));
      shadow = shadowCalculation(fragPosLightSpace, 0, input);
      for (int s = 0; (s < 4); s++) {
        if ((s >= (globals.frameCount % 3))) {
          continue;
        }
        shadow += shadowCalculation(
            cgl_float4_add(fragPosLightSpace,
                           make_float4((globals.noiseValues[(s % 16)] * 0.001),
                                       0.0, 0.0, 0.0)),
            (s + 1), input);
      }
      shadow /= 5.0;
    }
    Lo = cgl_float3_add(
        Lo, cgl_float3_mul_scalar(
                cgl_float3_mul(
                    cgl_scalar_mul_float3(
                        (1.0 - shadow),
                        cgl_float3_add(cgl_float3_div_scalar(
                                           cgl_float3_mul(kD, albedo), PI),
                                       specular)),
                    radiance),
                NdotL));
  }
  float3 ambient = cgl_float3_mul_scalar(
      cgl_float3_mul(globals.scene.ambientLight, albedo), ao);
  float3 color = cgl_float3_add(ambient, Lo);
  color =
      cgl_float3_div(color, cgl_float3_add(color, make_float3(1.0, 1.0, 1.0)));
  color = cgl_float3_pow_vector(color, cgl_float3_splat((1.0 / 2.2)));
  output.color = make_float4(color.x, color.y, color.z,
                             (material.opacity * albedoValue.w));
  output.normalBuffer = cgl_float4_construct_float3_xyz_float(
      cgl_float3_add_scalar(cgl_float3_mul_scalar(worldNormal, 0.5), 0.5), 1.0);
  output.positionBuffer = make_float4(
      input.worldPosition.x, input.worldPosition.y, input.worldPosition.z, 1.0);
  output.depth = (input.clipPosition.z / input.clipPosition.w);
  return output;
}

__global__ void compute_main() {
  int2 texCoord = make_int2((blockIdx.x * blockDim.x + threadIdx.x),
                            (blockIdx.y * blockDim.y + threadIdx.y));
  float2 screenSize = globals.screenSize;
  if (((texCoord.x >= int(screenSize.x)) ||
       (texCoord.y >= int(screenSize.y)))) {
    return;
  }
  float2 uv = cgl_float2_div(make_float2(texCoord.x, texCoord.y), screenSize);
  float4 color = make_float4(0.0, 0.0, 0.0, 0.0);
  float totalWeight = 0.0;
  float2 direction = cgl_float2_sub(make_float2(0.5, 0.5), uv);
  float len = cgl_float2_length(direction);
  direction = cgl_float2_normalize(direction);
  for (int i = 0; (i < 32); i++) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    float t = (float(i) / 32.0);
    float2 pos = cgl_float2_add(
        uv, cgl_float2_mul_scalar(
                cgl_float2_mul_scalar(cgl_float2_mul_scalar(direction, t), len),
                0.1));
    float noise =
        fbm(cgl_float3_construct_float2_xy_float(
                cgl_float2_mul_scalar(pos, 10.0), (globals.scene.time * 0.05)),
            4, 2.0, 0.5);
    float weight = (1.0 - t);
    weight = (weight * weight);
    float3 noiseColor = make_float3(
        (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 0.0)))),
        (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 2.0)))),
        (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 4.0)))));
    float3 __crossgl_swizzle_value_2 =
        cgl_float3_mul_scalar(noiseColor, weight);
    color.x = (color.x + __crossgl_swizzle_value_2.x);
    color.y = (color.y + __crossgl_swizzle_value_2.y);
    color.z = (color.z + __crossgl_swizzle_value_2.z);
    totalWeight += weight;
    direction = (float2x2(cosf((t * 3.0)), -sinf((t * 3.0)), sinf((t * 3.0)),
                          cosf((t * 3.0))) *
                 direction);
  }
  float __crossgl_swizzle_value_3 = totalWeight;
  color.x = (color.x / __crossgl_swizzle_value_3);
  color.y = (color.y / __crossgl_swizzle_value_3);
  color.z = (color.z / __crossgl_swizzle_value_3);
  color.w = 1.0;
  float vignette =
      (1.0 - cgl_smoothstep_float(
                 0.5, 1.0,
                 (cgl_float2_length(cgl_float2_sub_scalar(uv, 0.5)) * 1.5)));
  float __crossgl_swizzle_value_4 = vignette;
  color.x = (color.x * __crossgl_swizzle_value_4);
  color.y = (color.y * __crossgl_swizzle_value_4);
  color.z = (color.z * __crossgl_swizzle_value_4);
  surf2Dwrite(color, outputImage, texCoord.x * sizeof(float4), texCoord.y);
}
