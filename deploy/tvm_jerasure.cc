#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void tvm_ec_bitmatrix_multiply(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  DLTensor* x;
  DLTensor* y;
  DLTensor* z;
  int ndim = 2; // # of dimension of the array
  int dtype_code = kDLUInt;
  int dtype_bits = 8;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape_A[2] = {m*w, k*w};
  int64_t shape_B[2] = {k*w, packetsize};
  int64_t shape_C[2] = {m*w, packetsize};

  char* env_schedule = std::getenv("TVMEC_SCHEDULE_PATH");
  ICHECK(env_schedule != nullptr);
  std::string file_name = "/P_" + std::to_string(m) + "_n_" + std::to_string(packetsize) + "_D_" + std::to_string(k) + ".so";
  tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(env_schedule + file_name);
  tvm::runtime::PackedFunc f = mod.GetFunction("default_function");
  ICHECK(f != nullptr);

  // LOG(INFO) << "Shape: " << k << " " << m << " " << w << " " << packetsize;

  TVMArrayAlloc(shape_A, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape_B, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  TVMArrayAlloc(shape_C, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &z);

  // prepare encoding bitmatrix
  for (int i = 0; i < shape_A[0]; ++i) {
    for (int j = 0; j < shape_A[1]; ++j) {
      if (bitmatrix[i*k*w+j] != 0)
        static_cast<uint8_t*>(x->data)[i*k*w+j] = static_cast<uint8_t>(~0);
      else
        static_cast<uint8_t*>(x->data)[i*k*w+j] = static_cast<uint8_t>(0);
    }
  }
  // LOG(INFO) << "encoding bitmatrix init done";

  for (int i = 0; i < k; ++i) {
    std::memcpy(static_cast<uint8_t*>(y->data)+i*w*packetsize, data_ptrs[i], w*packetsize*sizeof(uint8_t));
  }
  // LOG(INFO) << "data matrix init done";

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y, z);

  // LOG(INFO) << "Finish calculation";
  TVMArrayFree(x);
  TVMArrayFree(y);

  for (int i = 0; i < m; ++i) {
    std::memcpy(coding_ptrs[i], static_cast<uint8_t*>(z->data)+i*w*packetsize, w*packetsize*sizeof(uint8_t));
  }
  // LOG(INFO) << "output written";
  TVMArrayFree(z);
}

#ifdef __cplusplus
}
#endif