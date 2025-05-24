// test5.cu
extern "C" __global__ void kernel5(int* a) {
    int tid = threadIdx.x;
    if (tid < 8) {
        a[tid] = 4;
    } else {
        a[tid] = 5;
    }
}
