#ifndef CU_LIB_HELPERS_CUSPLT_HELPER
#define CU_LIB_HELPERS_CUSPLT_HELPER

#include "cuda_helper.h"
#include "cusparseLt.h"

using fp16_type = uint16_t;

enum DenseLayout {
    DENSE_ROW_MAJOR,
    DENSE_COL_MAJOR
};

template<DenseLayout LayoutB=DENSE_COL_MAJOR, DenseLayout LayoutC=DENSE_COL_MAJOR>
struct CusparseLtSpmmaProblem 
{
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA;
    cusparseLtMatDescriptor_t matB;
    cusparseLtMatDescriptor_t matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream;

    float alpha;
    float beta;
    int length_m;
    int length_n;
    int length_k;
    fp16_type* dA;
    fp16_type* dB;
    fp16_type* dC;
    fp16_type* dA_compressed=nullptr;
    void* workspace=nullptr;

    CusparseLtSpmmaProblem(
        int length_m_, int length_n_, int length_k_, fp16_type* dA_, fp16_type* dB_, fp16_type* dC_,
        cudaStream_t stream_=nullptr, float alpha_=1.0f, float beta_=0.0f)
    : length_m(length_m_), length_n(length_n_), length_k(length_k_), dA(dA_), dB(dB_), dC(dC_),
    stream(stream_), alpha(alpha_), beta(beta_)
    {
        int align = 16;
        cusparseLtInit(&handle);
        cusparseLtStructuredDescriptorInit(&this->handle, &this->matA, length_m_, length_k_, 
                                    length_k_, // ldA
                                    align, // alignment 
                                    CUDA_R_16F, // type
                                    CUSPARSE_ORDER_ROW,  // order
                                    CUSPARSELT_SPARSITY_50_PERCENT // sparsity
                                    ); 
        cusparseLtDenseDescriptorInit(&this->handle, &this->matB, length_k_, length_n_, 
                                    (LayoutB==DENSE_COL_MAJOR ? length_k_ : length_n_), // ldB
                                    align, // alignment 
                                    CUDA_R_16F, // type
                                    (LayoutB==DENSE_COL_MAJOR ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW) // order
                                    );
        cusparseLtDenseDescriptorInit(&this->handle, &this->matC, length_m_, length_n_,
                                    (LayoutC==DENSE_COL_MAJOR ? length_m_ : length_n_), // ldB
                                    align, // alignment 
                                    CUDA_R_16F, // type
                                    (LayoutC==DENSE_COL_MAJOR ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW) // order
                                    );
        cusparseLtMatmulDescriptorInit(&this->handle, &this->matmul, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &this->matA, &this->matB, &this->matC, &this->matC, 
                                    CUSPARSE_COMPUTE_16F);
        cusparseLtMatmulAlgSelectionInit(&this->handle, &this->alg_sel, &this->matmul,
                                    CUSPARSELT_MATMUL_ALG_DEFAULT);
        int alg = 0;
        cusparseLtMatmulAlgSetAttribute(&this->handle, &this->alg_sel,
                                    CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                    &alg, sizeof(alg));
        size_t workspace_size, compressed_size;
        cusparseLtMatmulGetWorkspace(&this->handle, &this->alg_sel, &workspace_size);
        cusparseLtMatmulPlanInit(&this->handle, &this->plan, &this->matmul, &this->alg_sel, workspace_size);
        /// prepare workspace
        CUDA_CHECK(cudaMalloc((void**) &this->workspace, workspace_size));

        /// prune A
        cusparseLtSpMMAPrune(&this->handle, &this->matmul, dA_, dA_, CUSPARSELT_PRUNE_SPMMA_TILE, stream_);
        int *d_is_valid;
        CUDA_CHECK(cudaMalloc((void**)&d_is_valid, sizeof(d_is_valid)));

        cusparseLtSpMMAPruneCheck(&this->handle, &this->matmul, dA_, d_is_valid, stream_);
        int is_valid;
        CUDA_CHECK( cudaMemcpyAsync(&is_valid, d_is_valid, sizeof(d_is_valid),
                                cudaMemcpyDeviceToHost, stream_) )
        CUDA_CHECK( cudaStreamSynchronize(stream_) )
        if (is_valid != 0) {
            std::printf("!!!! The matrix has been pruned in a wrong way. "
                        "cusparseLtMatmul will not provided correct results\n");
        }
        /// compress A
        cusparseLtSpMMACompressedSize(&this->handle, &this->plan, &compressed_size);
        CUDA_CHECK(cudaMalloc((void**) &this->dA_compressed, compressed_size));
        cusparseLtSpMMACompress(&this->handle, &this->plan, dA_, dA_compressed, stream_);
    }

    ~CusparseLtSpmmaProblem() {
        if (this->workspace != nullptr)
        {
            CUDA_CHECK(cudaFree(this->workspace));
        }
        if (this->dA_compressed != nullptr) 
        {
            CUDA_CHECK(cudaFree(this->dA_compressed));
        }
        cusparseLtMatmulPlanDestroy(&this->plan);
        cusparseLtDestroy(&this->handle);
    }

    void _bareRun() {
        cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dC, workspace, nullptr, 0) ;
    }
    cudaError_t run() {
        _bareRun();
        cudaDeviceSynchronize();
        return cudaGetLastError();
    } 

    float benchmark(int warmupIters=10, int repeatIters=100) {
        GpuTimer gpuTimer;
        for (int iteration=0; iteration < warmupIters; iteration++) {
            _bareRun();
        }
        CUDA_CHECK( cudaPeekAtLastError());
        CUDA_CHECK( cudaDeviceSynchronize());

        gpuTimer.start();
        for (int iteration=0; iteration < repeatIters; iteration++) {
            _bareRun();
        }
        gpuTimer.stop();

        // // wipe memory
        // CUDA_CHECK( cudaFree(workspace) );
        return (gpuTimer.elapsed_msecs()/((float)repeatIters));
    }
};


#endif // CU_LIB_HELPERS_CUSPLT_HELPER