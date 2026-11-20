CUDA and HIP output place CrossGL `device` variables in device memory. They
used to emit ordinary host globals that kernels could not access (#4560, #4562).
