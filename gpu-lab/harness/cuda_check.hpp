#pragma once

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace gpu_lab {

inline void cuda_check(CUresult result, const char *expression) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = "unknown";
  const char *message = "unknown";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(expression) + ": " + name + ": " +
                           message);
}

} // namespace gpu_lab

#define GPU_LAB_CU_CHECK(call) ::gpu_lab::cuda_check((call), #call)
