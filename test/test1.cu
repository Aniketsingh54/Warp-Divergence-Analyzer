// test1.cu
extern "C" __global__ void kernel1(int* a) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        a[tid] = 1;
    } else {
        a[tid] = 2;
    }
}
