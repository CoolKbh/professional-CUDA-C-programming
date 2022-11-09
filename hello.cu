#include<stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello world from GPU!");
}

int main(void){
    //hello from CPU
    printf("Hello world from CPU!\n");

    //hello from GPU
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}