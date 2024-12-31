
// Description: hgemm main：测试GEMM的主程序，使用 CUDA 和 OpenMP，并通过 GFlags 进行参数管理

#include "gflags/gflags.h"
#include "omp.h"
#include "utils.cuh"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// 使用 GFlags 定义了一系列参数
DEFINE_int32(M, 16, "M");
DEFINE_int32(N, 8, "N");
DEFINE_int32(K, 16, "K");
DEFINE_bool(FP8E4M3, false, "test FP8E4M3 using mma");
DEFINE_bool(FP8E5M2, false, "test FP8E5M2 using mma");
DEFINE_bool(TF32, false, "test TF32 using mma");
DEFINE_bool(FP16, false, "test FP16 using mma");
DEFINE_string(A_filename, "./matrix_A.txt", "matrix_A.txt file path");
DEFINE_string(B_filename, "./matrix_B.txt", "matrix_B.txt file path");
DEFINE_string(C_filename, "./matrix_C.txt", "matrix_C.txt file path");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");


template <typename data_type_A, typename data_type_B, typename data_type_C>
void run(bool FLAGS_FP8E5M2, bool FLAGS_FP8E4M3, bool FLAGS_TF32, bool FLAGS_FP16){
    //host matrices
    data_type_A *hA = NULL;
    data_type_B *hB = NULL;
    data_type_C *hC = NULL; 
    data_type_C *hC_ref = NULL; 
    hA = (data_type_A *) malloc(sizeof(data_type_A) * FLAGS_M * FLAGS_K);
    hB = (data_type_B *) malloc(sizeof(data_type_B) * FLAGS_K * FLAGS_N);
    hC = (data_type_C *) malloc(sizeof(data_type_C) * FLAGS_M * FLAGS_N);
    hC_ref = (data_type_C *) malloc(sizeof(data_type_C) * FLAGS_M * FLAGS_N);
    initialize_matrices<data_type_A, data_type_B, data_type_C>(FLAGS_A_filename, hA, FLAGS_B_filename, hB, hC, FLAGS_M, FLAGS_N, FLAGS_K);
    copy_matrix<data_type_C>(hC, hC_ref, FLAGS_M * FLAGS_N);
    
    //device matrices
    data_type_A *dA = NULL;
    data_type_B *dB = NULL;
    data_type_C *dC = NULL; 
    data_type_C *dC_ref = NULL; 
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **) &dA, sizeof(data_type_A) * FLAGS_M * FLAGS_K));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **) &dB, sizeof(data_type_B) * FLAGS_K * FLAGS_N));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **) &dC, sizeof(data_type_C) * FLAGS_M * FLAGS_N));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **) &dC_ref, sizeof(data_type_C) * FLAGS_M * FLAGS_N));

    // copy host to device
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dA, hA, sizeof(data_type_A) * FLAGS_M * FLAGS_K, cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dB, hB, sizeof(data_type_B) * FLAGS_K * FLAGS_N, cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dC, hC, sizeof(data_type_C) * FLAGS_M * FLAGS_N, cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dC_ref, hC_ref, sizeof(float) * FLAGS_M * FLAGS_N, cudaMemcpyHostToDevice));
    
    // 调用mma Gemm
    if (FLAGS_FP8E5M2) {
        cublas_matmul_fp8<data_type_A, data_type_B, data_type_C>(dA, dB, dC_ref, FLAGS_M, FLAGS_N, FLAGS_K);      // cuBLAS
        launch_fp8_e5m2<data_type_A, data_type_B, data_type_C>(dA, dB, dC, FLAGS_M, FLAGS_N, FLAGS_K); // user define
        cudaDeviceSynchronize();
    }else if (FLAGS_FP8E4M3) {
        cublas_matmul_fp8<data_type_A, data_type_B, data_type_C>(dA, dB, dC_ref, FLAGS_M, FLAGS_N, FLAGS_K);      // cuBLAS
        launch_fp8_e4m3<data_type_A, data_type_B, data_type_C>(dA, dB, dC, FLAGS_M, FLAGS_N, FLAGS_K); // user define
        cudaDeviceSynchronize();
    }else if (FLAGS_TF32) {
        cublas_matmul_tf32<data_type_A, data_type_B, data_type_C>(dA, dB, dC_ref, FLAGS_M, FLAGS_N, FLAGS_K);      // cuBLAS
        launch_tf32<data_type_A, data_type_B, data_type_C>(dA, dB, dC, FLAGS_M, FLAGS_N, FLAGS_K); // user define
        cudaDeviceSynchronize();
    }else if (FLAGS_FP16) {
        cublas_matmul_fp16<data_type_A, data_type_B, data_type_C>(dA, dB, dC_ref, FLAGS_M, FLAGS_N, FLAGS_K);      // cuBLAS
        launch_fp16<data_type_A, data_type_B, data_type_C>(dA, dB, dC, FLAGS_M, FLAGS_N, FLAGS_K); // user define
        cudaDeviceSynchronize();
    }

    cudaMemcpy(hC, dC, sizeof(data_type_C) * FLAGS_M * FLAGS_N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_ref, dC_ref, sizeof(data_type_C) * FLAGS_M * FLAGS_N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check Result
    saveMatrixToFile<data_type_C>("./matrix_cublas.txt", hC_ref, FLAGS_M * FLAGS_N);
    saveMatrixToFile<data_type_C>("./matrix_C.txt", hC, FLAGS_M * FLAGS_N);
    verify_matrix<data_type_C>(hC_ref, hC, FLAGS_M * FLAGS_N);

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    HLOG("Done");

    // 释放CPU和GPU空间
    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
}

int main(int argc, char *argv[]) {
    
    // 获取参数
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    // 打印设备信息
    HGEMM_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));
    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    HLOG("CUDA HGEMM start on the %u-th GPU: %s", FLAGS_gpu_rank, dev_prop.name);
    int driver_version = 0;
    int runtime_version = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    HGEMM_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    HLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);       
    HLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor); 
    HLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);

    // 输出矩阵大小
    HLOG("A (%u x %u) * B (%u x %u) = C (%u x %u)", FLAGS_M, FLAGS_K, FLAGS_K, FLAGS_N, FLAGS_M, FLAGS_N);
    
    // 打印数据类型
    if (FLAGS_FP8E5M2) {
        HLOG("Selected data type: __nv_fp8_e5m2 " );
    } else if (FLAGS_FP8E4M3) {
        HLOG("Selected data type: __nv_fp8_e4m3 " );
    } else if (FLAGS_TF32) {
        HLOG("Selected data type: float (TF32) " );
    } else if (FLAGS_FP16) {
        HLOG("Selected data type: half " );
    } else {
        HLOG("No specific data type selected. Default: half" );
    }

    // 定义对应的数据类型并且调用run
    if(FLAGS_FP8E5M2) {

        using data_type_A =  __nv_fp8_e5m2;
        using data_type_B =  __nv_fp8_e4m3;
        using data_type_C =  float;
        run<data_type_A, data_type_B, data_type_C>(true, false, false, false);

    }else if(FLAGS_FP8E4M3){

        using data_type_A =  __nv_fp8_e4m3;
        using data_type_B =  __nv_fp8_e4m3;
        using data_type_C =  float;
        run<data_type_A, data_type_B, data_type_C>(false, true, false, false);

    }else if(FLAGS_TF32){

        using data_type_A =  float;
        using data_type_B =  float;
        using data_type_C =  float;
        run<data_type_A, data_type_B, data_type_C>(false, false, true, false);

    }else if(FLAGS_FP16){

        using data_type_A =  half;
        using data_type_B =  half;
        using data_type_C =  float;
        run<data_type_A, data_type_B, data_type_C>(false, false, false, true);

    }

    return 0;
}

