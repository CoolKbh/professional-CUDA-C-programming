#include<stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello world from GPU thread %d!\n", threadIdx.x);
}

int main(void){
    //hello from CPU
    printf("Hello world from CPU!\n");

    //hello from GPU
    helloFromGPU<<<2, 10>>>();
    cudaDeviceSynchronize();
    //cudaDeviceReset();
    return 0;
}
