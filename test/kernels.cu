extern "C" {

__global__ void kernel_if_else(int *data) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) data[tid] = tid * 2;
    else data[tid] = tid * 3;
}

__global__ void kernel_nested_branch(int *data) {
    int tid = threadIdx.x;
    if (tid < 16) {
        if (tid % 4 == 0) data[tid] = 10;
        else data[tid] = 20;
    } else {
        data[tid] = 30;
    }
}

__shared__ int flag;

__global__ void kernel_shared_flag(int *data) {
    int tid = threadIdx.x;
    if (tid == 0) flag = 1;
    __syncthreads();

    if (flag == 1) data[tid] = tid + 100;
    else data[tid] = tid + 200;
}

} // extern "C"
