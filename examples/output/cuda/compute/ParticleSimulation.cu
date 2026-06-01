#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline float cgl_float2_dot(float2 lhs, float2 rhs) {
  return (lhs.x * rhs.x) + (lhs.y * rhs.y);
}

__device__ inline float cgl_fract_float(float value) {
  return value - floorf(value);
}

__device__ inline float cgl_float3_dot(float3 lhs, float3 rhs) {
  return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

__device__ inline float3 cgl_float3_sin(float3 value) {
  return make_float3(sinf(value.x), sinf(value.y), sinf(value.z));
}

__device__ inline float3 cgl_float3_mul_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x * rhs), (lhs.y * rhs), (lhs.z * rhs));
}

__device__ inline float3 cgl_float3_fract(float3 value) {
  return make_float3(cgl_fract_float(value.x), cgl_fract_float(value.y),
                     cgl_fract_float(value.z));
}

__device__ inline float3 cgl_scalar_mul_float3(float lhs, float3 rhs) {
  return make_float3((lhs * rhs.x), (lhs * rhs.y), (lhs * rhs.z));
}

__device__ inline float3 cgl_scalar_add_float3(float lhs, float3 rhs) {
  return make_float3((lhs + rhs.x), (lhs + rhs.y), (lhs + rhs.z));
}

__device__ inline float3 cgl_float3_sub(float3 lhs, float3 rhs) {
  return make_float3((lhs.x - rhs.x), (lhs.y - rhs.y), (lhs.z - rhs.z));
}

__device__ inline float cgl_float3_length(float3 value) {
  return sqrtf((value.x * value.x) + (value.y * value.y) + (value.z * value.z));
}

__device__ inline float3 cgl_float3_normalize(float3 value) {
  float inv_length = 1.0f / cgl_float3_length(value);
  return make_float3((value.x * inv_length), (value.y * inv_length),
                     (value.z * inv_length));
}

__device__ inline float3 cgl_float3_div_scalar(float3 lhs, float rhs) {
  return make_float3((lhs.x / rhs), (lhs.y / rhs), (lhs.z / rhs));
}

__device__ inline float3 cgl_float3_add(float3 lhs, float3 rhs) {
  return make_float3((lhs.x + rhs.x), (lhs.y + rhs.y), (lhs.z + rhs.z));
}

__device__ inline float4 cgl_float4_mix_scalar(float4 x, float4 y, float a) {
  return make_float4((x.x + ((y.x - x.x) * a)), (x.y + ((y.y - x.y) * a)),
                     (x.z + ((y.z - x.z) * a)), (x.w + ((y.w - x.w) * a)));
}

__device__ inline float3 cgl_float3_mix_scalar(float3 x, float3 y, float a) {
  return make_float3((x.x + ((y.x - x.x) * a)), (x.y + ((y.y - x.y) * a)),
                     (x.z + ((y.z - x.z) * a)));
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

struct Particle {
  float3 position;
  float3 velocity;
  float3 acceleration;
  float mass;
  float lifetime;
  float4 color;
  int type;
  bool active;
};

struct PhysicsConstants {
  float gravity;
  float damping;
  float timestep;
  float collision_radius;
  float3 world_bounds_min;
  float3 world_bounds_max;
  int max_particles;
  float attraction_strength;
};

struct SimulationState {
  int active_particle_count;
  int frame_number;
  float total_time;
  float3 attractor_position;
};

struct ParticleBuffer {
  Particle particles[4096];
};

struct AtomicCounters {
  int collision_count;
  int active_count;
  int spawn_count;
};

const int MAX_PARTICLES = 4096;

const int WORKGROUP_SIZE = 64;

const float PI = 3.14159265359;

const float3 GRAVITY_VECTOR = make_float3(0.0, -9.81, 0.0);

__constant__ PhysicsConstants physics;

__constant__ SimulationState sim_state;

ParticleBuffer particle_buffer;

AtomicCounters counters;

__device__ float random(float2 st) {
  return cgl_fract_float((sinf(cgl_float2_dot(make_float2(st.x, st.y),
                                              make_float2(12.9898, 78.233))) *
                          43758.5453123));
}

__device__ float3 random3(float3 seed) {
  float3 p =
      make_float3(cgl_float3_dot(seed, make_float3(127.1, 311.7, 74.7)),
                  cgl_float3_dot(seed, make_float3(269.5, 183.3, 246.1)),
                  cgl_float3_dot(seed, make_float3(113.5, 271.9, 124.6)));
  return cgl_scalar_add_float3(
      -1.0, cgl_scalar_mul_float3(2.0, cgl_float3_fract(cgl_float3_mul_scalar(
                                           cgl_float3_sin(p), 43758.5453123))));
}

__device__ float3 calculateAttraction(float3 position, float3 attractor_pos,
                                      float strength) {
  float3 direction = cgl_float3_sub(attractor_pos, position);
  float distance = cgl_float3_length(direction);
  if ((distance < 0.001)) {
    return make_float3(0.0, 0.0, 0.0);
  }
  float force = (strength / ((distance * distance) + 0.1));
  return cgl_float3_mul_scalar(cgl_float3_normalize(direction), force);
}

__device__ bool checkCollision(float3 pos1, float3 pos2, float radius) {
  return (cgl_float3_length(cgl_float3_sub(pos1, pos2)) < (radius * 2.0));
}

__device__ float3 resolveCollision(float3 pos1, float3 vel1, float3 pos2,
                                   float3 vel2, float mass1, float mass2) {
  float3 relative_pos = cgl_float3_sub(pos1, pos2);
  float3 relative_vel = cgl_float3_sub(vel1, vel2);
  float distance = cgl_float3_length(relative_pos);
  if ((distance < 0.001)) {
    return vel1;
  }
  float3 normal = cgl_float3_div_scalar(relative_pos, distance);
  float relative_speed = cgl_float3_dot(relative_vel, normal);
  if ((relative_speed > 0.0)) {
    return vel1;
  }
  float impulse = ((2.0 * relative_speed) / (mass1 + mass2));
  return cgl_float3_sub(vel1, cgl_scalar_mul_float3((impulse * mass2), normal));
}

extern "C" __global__ void compute_main() {
  __shared__ float3 shared_positions[WORKGROUP_SIZE];
  __shared__ float3 shared_velocities[WORKGROUP_SIZE];
  __shared__ int shared_types[WORKGROUP_SIZE];
  __shared__ bool shared_active[WORKGROUP_SIZE];
  int particle_id = int((blockIdx.x * blockDim.x + threadIdx.x));
  int local_id = int(threadIdx.x);
  if ((particle_id >= physics.max_particles)) {
    return;
  }
  Particle particle = particle_buffer.particles[particle_id];
  if (!particle.active) {
    shared_active[local_id] = false;
    __syncthreads();
    return;
  }
  shared_positions[local_id] = particle.position;
  shared_velocities[local_id] = particle.velocity;
  shared_types[local_id] = particle.type;
  shared_active[local_id] = true;
  __syncthreads();
  float3 force = make_float3(0.0, 0.0, 0.0);
  if ((particle.type == 0)) {
    force = cgl_float3_add(
        force, cgl_float3_mul_scalar(GRAVITY_VECTOR, particle.mass));
  }
  if ((particle.type != 2)) {
    force =
        cgl_float3_add(force, calculateAttraction(particle.position,
                                                  sim_state.attractor_position,
                                                  physics.attraction_strength));
  }
  int collision_count = 0;
  for (int i = 0; (i < WORKGROUP_SIZE); i++) {
    if (((i == local_id) || !shared_active[i])) {
      continue;
    }
    if (checkCollision(particle.position, shared_positions[i],
                       physics.collision_radius)) {
      collision_count++;
      particle.velocity = resolveCollision(
          particle.position, particle.velocity, shared_positions[i],
          shared_velocities[i], particle.mass, 1.0);
      particle.color = cgl_float4_mix_scalar(
          particle.color, make_float4(1.0, 0.0, 0.0, 1.0), 0.1);
    }
    if (((shared_types[i] == 1) && (particle.type == 0))) {
      float3 attraction =
          calculateAttraction(particle.position, shared_positions[i], 0.5);
      force = cgl_float3_add(force, attraction);
    }
  }
  particle.acceleration = cgl_float3_div_scalar(force, particle.mass);
  particle.velocity = cgl_float3_add(
      particle.velocity,
      cgl_float3_mul_scalar(particle.acceleration, physics.timestep));
  particle.velocity = cgl_float3_mul_scalar(particle.velocity, physics.damping);
  particle.position = cgl_float3_add(
      particle.position,
      cgl_float3_mul_scalar(particle.velocity, physics.timestep));
  if (((particle.position.x < physics.world_bounds_min.x) ||
       (particle.position.x > physics.world_bounds_max.x))) {
    particle.velocity.x *= -0.8;
    particle.position.x =
        fmaxf(physics.world_bounds_min.x,
              fminf(physics.world_bounds_max.x, particle.position.x));
  }
  if (((particle.position.y < physics.world_bounds_min.y) ||
       (particle.position.y > physics.world_bounds_max.y))) {
    particle.velocity.y *= -0.8;
    particle.position.y =
        fmaxf(physics.world_bounds_min.y,
              fminf(physics.world_bounds_max.y, particle.position.y));
  }
  if (((particle.position.z < physics.world_bounds_min.z) ||
       (particle.position.z > physics.world_bounds_max.z))) {
    particle.velocity.z *= -0.8;
    particle.position.z =
        fmaxf(physics.world_bounds_min.z,
              fminf(physics.world_bounds_max.z, particle.position.z));
  }
  particle.lifetime -= physics.timestep;
  if ((particle.lifetime <= 0.0)) {
    particle.active = false;
    atomicAdd(&counters.active_count, -1);
  }
  float speed = cgl_float3_length(particle.velocity);
  float3 __crossgl_swizzle_value_0 = cgl_float3_mix_scalar(
      make_float3(0.2, 0.4, 1.0), make_float3(1.0, 0.4, 0.2),
      fmaxf(0.0, fminf(1.0, (speed / 10.0))));
  particle.color.x = __crossgl_swizzle_value_0.x;
  particle.color.y = __crossgl_swizzle_value_0.y;
  particle.color.z = __crossgl_swizzle_value_0.z;
  particle_buffer.particles[particle_id] = particle;
  if ((collision_count > 0)) {
    atomicAdd(&counters.collision_count, collision_count);
  }
}

extern "C" __global__ void spawn() {
  if ((counters.active_count >= physics.max_particles)) {
    return;
  }
  for (int i = 0; (i < physics.max_particles); i++) {
    if (!particle_buffer.particles[i].active) {
      Particle new_particle;
      float3 random_offset =
          random3(make_float3(i, sim_state.frame_number, sim_state.total_time));
      new_particle.position =
          cgl_float3_add(sim_state.attractor_position,
                         cgl_float3_mul_scalar(random_offset, 2.0));
      new_particle.velocity = cgl_float3_mul_scalar(
          random3(make_float3((i + 1000), sim_state.frame_number,
                              sim_state.total_time)),
          5.0);
      new_particle.acceleration = make_float3(0.0, 0.0, 0.0);
      new_particle.mass =
          (1.0 + (random(make_float2(i, sim_state.frame_number)) * 2.0));
      new_particle.lifetime =
          (10.0 +
           (random(make_float2((i + 500), sim_state.frame_number)) * 20.0));
      new_particle.color = make_float4(
          random(make_float2((i + 100), sim_state.frame_number)),
          random(make_float2((i + 200), sim_state.frame_number)),
          random(make_float2((i + 300), sim_state.frame_number)), 1.0);
      new_particle.type =
          int((random(make_float2((i + 400), sim_state.frame_number)) * 3.0));
      new_particle.active = true;
      particle_buffer.particles[i] = new_particle;
      atomicAdd(&counters.active_count, 1);
      atomicAdd(&counters.spawn_count, 1);
      break;
    }
  }
}
