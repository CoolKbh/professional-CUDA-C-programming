#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size){
    //generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float)( rand() & 0xFF) / 10.0f;
    }
}

int main() {
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcoy(d_B, h_B, nBytes, cudaMemcpyHostToDevice):

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArrayOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
