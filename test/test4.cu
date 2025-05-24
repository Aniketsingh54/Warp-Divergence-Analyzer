// test4.cu
extern "C" __global__ void kernel4(int* a) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int ntid = blockDim.x;
    int gid = bid * ntid + tid;

    if (gid % 2 == 0) {
        if (gid < 64) {
            a[gid] = 1;
        } else {
            a[gid] = 2;
        }
    } else {
        a[gid] = 3;
    }
}
