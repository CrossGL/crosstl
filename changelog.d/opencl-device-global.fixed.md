OpenCL C input imports a program-scope `global` variable as a CrossGL `device`
variable instead of refusing it, so CUDA and HIP output declare it `__device__`.
The other targets refuse it with the typed device-variable diagnostic (#4563).
