#include "../cusparseLt_helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

int main(int argc, char** argv)
{
    int length_m = atoi(argv[1]);
    int length_n = atoi(argv[2]);
    int length_k = atoi(argv[3]);

    bool isColMajorB = true; 
    bool isColMajorC = true;
    if (argc>4) {
        std::string rowstr = "row";
        if (argv[4]==rowstr) 
            isColMajorB = false;
        if (argv[5]==rowstr) 
            isColMajorC = false;
    }
    
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_A({length_m, length_k});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_B({length_k, length_n});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_C({length_m, length_n});
    // cutlass::reference::host::TensorFill(tensor_A.host_view(), hf(1));
    // cutlass::reference::host::TensorFill(tensor_B.host_view(), hf(1));
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_A.host_view(),
        1,
        cutlass::half_t(4),
        cutlass::half_t(-4),
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_B.host_view(),
        1,
        cutlass::half_t(4),
        cutlass::half_t(-4),
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(tensor_C.host_view());
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();

    fp16_type *dA = (fp16_type*)tensor_A.device_data();
    fp16_type *dB = (fp16_type*)tensor_B.device_data();
    fp16_type *dC = (fp16_type*)tensor_C.device_data();

    if (isColMajorB && isColMajorC) {
        CusparseLtSpmmaProblem<> problem(length_m, length_n, length_k, dA, dB, dC);
        CUDA_CHECK(problem.run());
        float dur = problem.benchmark();
        float gflop = (float)length_m/1e9*length_n*length_k*2;
        float tflops = gflop/dur;
        printf("SpMMA,%f\n", tflops);
    }
    else if (isColMajorB && ~isColMajorC) {
        CusparseLtSpmmaProblem<DENSE_COL_MAJOR, DENSE_ROW_MAJOR> problem(length_m, length_n, length_k, dA, dB, dC);
        CUDA_CHECK(problem.run());
        float dur = problem.benchmark();
        float gflop = (float)length_m/1e9*length_n*length_k*2;
        float tflops = gflop/dur;
        printf("SpMMA,%f\n", tflops);
    }
    else if ((~isColMajorB) && isColMajorC) {
        CusparseLtSpmmaProblem<DENSE_ROW_MAJOR, DENSE_COL_MAJOR> problem(length_m, length_n, length_k, dA, dB, dC);
        CUDA_CHECK(problem.run());
        float dur = problem.benchmark();
        float gflop = (float)length_m/1e9*length_n*length_k*2;
        float tflops = gflop/dur;
        printf("SpMMA,%f\n", tflops);
    }
    else if ((~isColMajorB) && (~isColMajorC)) {
        CusparseLtSpmmaProblem<DENSE_ROW_MAJOR, DENSE_ROW_MAJOR> problem(length_m, length_n, length_k, dA, dB, dC);
        CUDA_CHECK(problem.run());
        float dur = problem.benchmark();
        float gflop = (float)length_m/1e9*length_n*length_k*2;
        float tflops = gflop/dur;
        printf("SpMMA,%f\n", tflops);       
    }
    return 0;
}