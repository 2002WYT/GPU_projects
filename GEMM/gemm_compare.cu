#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"cuBLAS error: %d\n",(int)s); std::exit(1);} } while(0)

template<int T=16>
__global__ void gemm_kernel(int m,int n,int k,const double* A,const double* B,double* C)
{
    __shared__ double As[T][T+1], Bs[T][T+1];
    int row=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x;
    double sum=0.0;

    for(int base=0;base<k;base+=T){
        int ac=base+threadIdx.x, br=base+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<m && ac<k)?A[row+ac*m]:0.0;
        Bs[threadIdx.y][threadIdx.x]=(br<k && col<n)?B[br+col*k]:0.0;
        __syncthreads();
#pragma unroll
        for(int p=0;p<T;++p) sum+=As[threadIdx.y][p]*Bs[p][threadIdx.x];
        __syncthreads();
    }
    if(row<m && col<n) C[row+col*m]=sum;
}

void custom_gemm(int m,int n,int k,const double* A,const double* B,double* C)
{
    dim3 block(16,16), grid((n+15)/16,(m+15)/16);
    gemm_kernel<<<grid,block>>>(m,n,k,A,B,C);
}

template<class F>
float benchmark(F launch,int repeat)
{
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    for(int i=0;i<5;++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;++i) launch();
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return ms/repeat;
}

int main(int argc,char** argv)
{
    int m=argc>1?std::atoi(argv[1]):1024;
    int n=argc>2?std::atoi(argv[2]):1024;
    int k=argc>3?std::atoi(argv[3]):1024;
    int repeat=argc>4?std::atoi(argv[4]):100;

    std::vector<double> A((size_t)m*k),B((size_t)k*n),C1((size_t)m*n),C2((size_t)m*n);
    for(size_t i=0;i<A.size();++i) A[i]=(static_cast<int>(i%100)-50)*0.01;
    for(size_t i=0;i<B.size();++i) B[i]=(static_cast<int>(i%80)-40)*0.01;

    double *dA,*dB,*dC1,*dC2;
    CUDA_CHECK(cudaMalloc(&dA,A.size()*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB,B.size()*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC1,C1.size()*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC2,C2.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,B.data(),B.size()*sizeof(double),cudaMemcpyHostToDevice));

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    const double alpha=1.0,beta=0.0;

    custom_gemm(m,n,k,dA,dB,dC1);
    CUBLAS_CHECK(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,
                            &alpha,dA,m,dB,k,&beta,dC2,m));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C1.data(),dC1,C1.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C2.data(),dC2,C2.size()*sizeof(double),cudaMemcpyDeviceToHost));

    double diff2=0.0, ref2=0.0;
    for(size_t i=0;i<C1.size();++i){
        double d=C1[i]-C2[i];
        diff2+=d*d; ref2+=C2[i]*C2[i];
    }
    double rel_err=std::sqrt(diff2)/(std::sqrt(ref2)+1e-30);

    float custom_ms=benchmark([&]{custom_gemm(m,n,k,dA,dB,dC1);},repeat);
    float cublas_ms=benchmark([&]{CUBLAS_CHECK(cublasDgemm(
        handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,dA,m,dB,k,&beta,dC2,m));},repeat);

    double ops=2.0*m*n*k;
    std::printf("m=%d n=%d k=%d repeat=%d\n",m,n,k,repeat);
    std::printf("custom : %8.3f ms  %8.2f GFLOPS\n",custom_ms,ops/(custom_ms*1e6));
    std::printf("cuBLAS : %8.3f ms  %8.2f GFLOPS\n",cublas_ms,ops/(cublas_ms*1e6));
    std::printf("relative error: %.3e\n",rel_err);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC1)); CUDA_CHECK(cudaFree(dC2));
}
