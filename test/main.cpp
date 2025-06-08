#include <iostream>
#include <cuda_runtime.h>

extern "C" {
    void kernel_if_else(int*);
    void kernel_nested_branch(int*);
    void kernel_shared_flag(int*);
}

int main() {
    const int N = 32;
    int *d_data;
    int h_data[N];

    cudaMalloc(&d_data, N * sizeof(int));

    kernel_if_else<<<1, N>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "kernel_if_else results:\n";
    for (int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << "\n";

    kernel_nested_branch<<<1, N>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "kernel_nested_branch results:\n";
    for (int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << "\n";

    kernel_shared_flag<<<1, N>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "kernel_shared_flag results:\n";
    for (int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << "\n";

    cudaFree(d_data);
    return 0;
}
