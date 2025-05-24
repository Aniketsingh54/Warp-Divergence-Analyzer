// test2.cu
extern "C" __global__ void kernel2(int* a) {
    int tid = threadIdx.x;
    a[tid] = tid;
}
