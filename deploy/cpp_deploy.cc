/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <iostream>

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  ICHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* x;
  DLTensor* y;
  DLTensor* z;
  int ndim = 2;
  int dtype_code = kDLUInt;
  int dtype_bits = 8;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape_A[2] = {16, 32};
  int64_t shape_B[2] = {32, 65536};
  int64_t shape_C[2] = {16, 65536};
  TVMArrayAlloc(shape_A, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape_B, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  TVMArrayAlloc(shape_C, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &z);
  for (int i = 0; i < shape_A[0]; ++i) {
    for (int j = 0; j < shape_A[1]; ++j) {
      static_cast<float*>(x->data)[i] = i;
    }
  }
  for (int i = 0; i < shape_B[0]; ++i) {
    for (int j = 0; j < shape_B[1]; ++j) {
      static_cast<float*>(x->data)[i] = i;
    }
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y, z);
  // Print out the output
  LOG(INFO) << "Finish verification...";
  TVMArrayFree(x);
  TVMArrayFree(y);
}

void tvm_ec_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  // tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("lib/P_2_n_4096_D_2_xl170.so");
  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  LOG(INFO) << "Mod load success";
  tvm::runtime::PackedFunc f = mod.GetFunction("default_function");
  ICHECK(f != nullptr);
  LOG(INFO) << "Function load success";

  DLTensor* x;
  DLTensor* y;
  DLTensor* z;
  int ndim = 2; // # of dimension of the array
  int dtype_code = kDLUInt;
  int dtype_bits = 8;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int blk_size = w*packetsize;
  int64_t shape_A[2] = {m*w, k*w};
  int64_t shape_B[2] = {k*w, blk_size};
  int64_t shape_C[2] = {m*w, blk_size};
  std::cout << m*w << " " << k*w << " " << blk_size << std::endl;
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
  LOG(INFO) << "encoding bitmatrix init done";

  for (int i = 0; i < shape_B[0]; ++i) {
    for (int j = 0; j < shape_B[1]; ++j) {
      static_cast<uint8_t*>(y->data)[i*blk_size+j] = data_ptrs[i][j];
    }
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y, z);
  // Print out the output
  LOG(INFO) << "Finish verification...";
  TVMArrayFree(x);
  TVMArrayFree(y);
}

void TestForJerasure(int k, int m, int w, int packetsize) {
  int *bitmatrix;
  char **data_ptrs, **coding_ptrs;
  int blk_size = packetsize*w;
  bitmatrix = talloc(int, k*w*m*w);
  data_ptrs = talloc(char *, k*w);
  LOG(INFO) << "Jerasure test start";

  // initialize data matrix
  for (int i = 0; i < k*w; i++) {
    data_ptrs[i] = talloc(char, blk_size);
    for (int j = 0; j < blk_size; j++) {
      data_ptrs[i][j] = static_cast<char>(i*blk_size+j);
    }
  }

  coding_ptrs = talloc(char *, m*w);
  for (int i = 0; i < m*w; i++) {
    coding_ptrs[i] = talloc(char, blk_size);
  }

  // initialize encoding bitmatrix
  for (int i = 0; i < m*w; i++) {
    for (int j = 0; j < k*w; j++) {
      bitmatrix[i*k*w+j] = static_cast<char>((i*k*w+j)%2);
    }
  }

  LOG(INFO) << "matrix init done";

  tvm_ec_bitmatrix_encode(k, m, w, bitmatrix, data_ptrs, coding_ptrs, 0, packetsize);
}

void DeploySingleOp() {
  // Normally we can directly
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("lib/test.so");
  LOG(INFO) << "Verify dynamic loading from test_addone_dll.so";
  Verify(mod_dylib, "default_function");
}

int main(void) {
  TestForJerasure(2, 2, 8, 512);
  return 0;
}