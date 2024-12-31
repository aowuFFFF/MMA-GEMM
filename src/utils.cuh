#include <stdio.h>
#include <fstream>
#include <sstream>
#include "kernel.cuh"
#include <cublasLt.h>
#include "common.h"
#include <cuda_fp8.h>
#include <type_traits>
#include <cstring> 
#include <limits> 

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

/*
=====================================
矩阵操作
=====================================
*/
template <typename data_type_C>
void verify_matrix(data_type_C *mat1, data_type_C *mat2, int N) {
    float m_max_diff = 0.0;
    float min_diff = 0.0;
    float diff = 0.0;
    const float EPS = 0.0000001;
    for (size_t i = 0; i < N; ++i) {     
        diff = static_cast<float>(std::abs(mat1[i] - mat2[i]));
        // diff = fabs((double) mat1[i] - (double) mat2[i]);
        m_max_diff = std::max(m_max_diff, diff);
        if(diff >= EPS)  min_diff = 1.0;
    }
    // 打印信息
    if (min_diff) {
        HLOG("[Matrix Verification] Warning: Differences detected exceeding tolerance (EPS = %.7f)", EPS);
    } else {
        HLOG("[Matrix Verification] Success: No significant differences detected (EPS = %.7f)", EPS);
    }

    HLOG("[Matrix Verification] Summary: Max Absolute Difference: %.7f | Total Elements: %d",
        m_max_diff, N);

}

template <typename data_type_C>
void copy_matrix(data_type_C *src, data_type_C *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        HLOG("copy failed at %d while there are %d elements in total.", i, N);
}

template <typename data_type_C>
void resetToZero(data_type_C* matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<data_type_C>(0);
    }
}

// 从文件读取数据
template <typename T>
void readFile(const std::string &filename , T* matrix, int rows, int cols) {
    int m_elem_num = rows * cols;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file: " + filename);
    }

    std::string line;
    size_t index = 0;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            if (index >= m_elem_num) {   
                throw std::out_of_range("Too many elements in the file.");
            }
            // 去掉十六进制字符串0x前缀
            if (value.find("0x") == 0) {
                value = value.substr(2); // 去掉0x前缀
            }

            // 根据传入的数据类型进行转换
            if constexpr (std::is_same<T, float>::value) {
                uint32_t byteValue = static_cast<uint32_t>(std::stoul(value, nullptr, 16));
                float floatValue;
                std::memcpy(&floatValue, &byteValue, sizeof(float));
                matrix[index++] = floatValue;
                // printf("matrix0: %f  \n", matrix[0]);

            } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {

                uint8_t byteValue = static_cast<uint8_t>(std::stoi(value, nullptr, 16));
                matrix[index++] = *reinterpret_cast<__nv_fp8_e5m2*>(&byteValue);

            } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {

                uint8_t byteValue = static_cast<uint8_t>(std::stoi(value, nullptr, 16));
                matrix[index++] = *reinterpret_cast<__nv_fp8_e4m3*>(&byteValue);

            } else if constexpr (std::is_same<T, half>::value) {
                uint16_t byteValue = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                half halfValue;
                std::memcpy(&halfValue, &byteValue, sizeof(half));
                matrix[index++] = halfValue;

            } else {
                HLOG("T is an unknown type");
            }

            
        }
    }
    if (index != m_elem_num) {
        throw std::runtime_error("Not enough data in file: " + filename);
    }

    file.close();
}

// 初始化矩阵
template <typename data_type_A, typename data_type_B, typename data_type_C>
void initialize_matrices(const std::string &filenameA, data_type_A* A, 
                         const std::string &filenameB, data_type_B* B, 
                        data_type_C* C, 
                         int M, int N, int K) {
    // 初始化 A、B 和 C 矩阵
    
    readFile<data_type_A>(filenameA, A, M, K);  // A 矩阵 M * K
    readFile<data_type_B>(filenameB, B, K, N);  // B 矩阵 K * N
    resetToZero(C, M * N);  // C 矩阵 M * N，设置为0
    
}

// 保存C矩阵
template <typename data_type_C>
void saveMatrixToFile(const std::string& filename, data_type_C *C, int N) {
    // 检查文件是否存在
    std::ifstream check_file(filename);
    if (check_file) { 
        check_file.close(); 
        std::remove(filename.c_str());
        // HLOG("Existing file and deleted.");
    }

    // 创建新文件并打开写入
    std::ofstream file(filename);

    // 遍历矩阵并输出到文件
    for (int i = 0; i < N; ++i) {
        // if(i == 1) HLOG("CMatrixResult[0] : %f ", C[0]);
        file << C[i];
        file << ",";
    }

    file.close(); 
    HLOG("Result Matrix C saved to %s", filename.c_str()); 
}

/*
=====================================
cutlas操作
=====================================
*/

// Cublas For fp8
template <typename T>
cudaDataType_t getCudaDataType() {
    if constexpr (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
        return CUDA_R_8F_E4M3;
    } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
        return CUDA_R_8F_E5M2;
    } else {
        std::cout << "The type is: Unknown" << std::endl;
        return CUDA_R_32F; // default type, can be changed
    }
}

template <typename data_type_A, typename data_type_B, typename data_type_C>
void cublas_matmul_fp8(data_type_A* a, data_type_B* b, data_type_C* c, int M, int N, int K) {
    
    // Get the cudaDataType_t for each data type
    cudaDataType_t dataTypeA, dataTypeB, dataTypeC;
    dataTypeA = getCudaDataType<data_type_A>();
    dataTypeB = getCudaDataType<data_type_B>();
    dataTypeC = getCudaDataType<data_type_C>();

    // 创建 cuBLAS 句柄
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // 创建矩阵描述符
    cublasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA, dataTypeA, K, M, K)); // A[K,M]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB, dataTypeB, K, N, K)); // B[K,N]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC, dataTypeC, M, N, M)); // C[M,N] in BF16
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matD, dataTypeC, M, N, M));// D[M,N] in FP8

    // 创建操作描述符
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // 设置操作属性 - FP8 格式需要 "TN"
    const int32_t transa = CUBLAS_OP_T;
    const int32_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t)));

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // 分配工作区
    size_t workspaceSize = 512 * 1024 * 1024;// 32MB 工作区
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // 查询最佳算法
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;

    // CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, matA, matB, matC, matD, preference, 1, &heuristicResult, &returnedResults));
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, matA, matB, matC, matD, preference, 1, &heuristicResult, &returnedResults);

    // 运行 cuBLAS 矩阵乘法
    CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc, &alpha, a, matA, b, matB, &beta, c, matC, c, matD, &heuristicResult.algo, workspace, workspaceSize, 0));

    CHECK_CUDA(cudaFree(workspace));
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(matA);
    cublasLtMatrixLayoutDestroy(matB);
    cublasLtMatrixLayoutDestroy(matC);
    cublasLtMatrixLayoutDestroy(matD);
    cublasLtDestroy(handle);
}

// Cublas For fp16
template <typename data_type_A, typename data_type_B, typename data_type_C>
void cublas_matmul_fp16(data_type_A *A, data_type_B *B, data_type_C *C, size_t M, size_t N, size_t K) {
    // 创建 cuBLAS 句柄
    cublasHandle_t handle = nullptr;
    HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    // 设置 cuBLAS 的数学模式为 CUBLAS_TENSOR_OP_MATH。该模式启用了张量运算优化
    HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // C = alpha * A * B + beta * C
    float alpha = 1.0;
    float beta = 1.0;

    HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                            CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    HGEMM_CHECK_CUBLAS_ERROR(cublasDestroy(handle));

}

// Cublas For TF32
template <typename data_type_A, typename data_type_B, typename data_type_C>
void cublas_matmul_tf32(data_type_A *A, data_type_B *B, data_type_C *C, size_t M, size_t N, size_t K) {   
    
    // 创建 cuBLAS 句柄
    cublasHandle_t handle = nullptr;
    HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    // 设置 cuBLAS 的数学模式为 CUBLAS_TENSOR_OP_MATH。该模式启用了张量运算优化
    HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // C = alpha * A * B + beta * C
    float alpha = 1.0;
    float beta = 1.0;
    
    if constexpr (std::is_same<data_type_A, float>::value) {
        HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, K, A,
                                            CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    
    HGEMM_CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

/*
=====================================
kernel操作
=====================================
*/
//调用对应数据类型kernel计算矩阵乘法
#define WARP_SIZE 32

// FP8:e5m2
template <typename AType, typename BType, typename CType>
void launch_fp8_e5m2(AType *A, BType *B, CType *C, int M, int N, int K) {
    dim3 block(WARP_SIZE);
    dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));

    mma_fp8_e5m2<AType, BType, CType><<<grid, block>>>(A, B, C, M, N, K);  
}

// FP8:e4m3
template <typename AType, typename BType, typename CType>
void launch_fp8_e4m3(AType *A, BType *B, CType *C, int M, int N, int K) {
    dim3 block(WARP_SIZE);
    dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));

    mma_fp8_e4m3<AType, BType, CType><<<grid, block>>>(A, B, C, M, N, K);
  
}

// TF32
template <typename AType, typename BType, typename CType>
void launch_tf32(AType *A, BType *B, CType *C, size_t M, size_t N, size_t K) {
    
    dim3 block(WARP_SIZE);
    dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));

    mma_tf32<AType, BType, CType><<<grid, block>>>(A, B, C, M, N, K);
}

// FP16
template <typename AType, typename BType, typename CType>
void launch_fp16(AType *A, BType *B, CType *C, size_t M, size_t N, size_t K) {
    
    dim3 block(WARP_SIZE);
    dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));

    mma_fp16<AType, BType, CType><<<grid, block>>>(A, B, C, M, N, K);
}
