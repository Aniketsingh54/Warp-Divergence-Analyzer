// test3.cu
extern "C" __global__ void kernel3(int* a) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        a[tid] = 10;
    } else if (tid < 16) {
        a[tid] = 20;
    } else {
        a[tid] = 30;
    }
}
