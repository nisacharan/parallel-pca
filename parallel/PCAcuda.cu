#include <cuda.h>
#include <bits/stdc++.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>



using namespace std;

#define BLOCK_SIZE 1
#define BLOCK_ROWS 8
#define TILE_DIM 32
#define threadsPerBlock 256

typedef struct {
    int width;
    int height;
    int stride; 
    double* elements;
} Matrix;



void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f ### ", name, row+1, col+1, Areg);
        }
    }
}


__global__ void MatTrans(Matrix odata, const Matrix idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width =32 * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
    //if(x<idata.height && y<idata.width){
       odata.elements[x*width + (y+j)] = idata.elements[(y+j)*width + x];
       //printf("I'm thread:%d, from block, %d\n",threadIdx.x,blockIdx.x);
       //printf("I have put %d, in place of %d\n",idata.elements[(y+j)*width + x],idata.elements[(y+j)*width + x]);
  //}
  }
}


//perfect for threads
__global__ void MatMeanKernel (const Matrix img, Matrix lineImg, int height, int width) 
{
  // height = 1024, width = 512
  int tidy = threadIdx.x + blockDim.x * blockIdx.x; 

  double sum = 0.0f; 
  double sumDiv = 0.0f; 

  if(tidy < height) { 

      for(int c = 0; c < width; c++) 
      { 
          sum += img.elements[tidy+width * c];
      }
      sumDiv = double(sum)/height;

      __syncthreads(); 
      for(int cc = 0; cc < width; cc++) 
      { 

          lineImg.elements[tidy+width * cc] = (img.elements[tidy+width * cc] - sumDiv);
      }

  }

  __syncthreads(); 
}

// Compute C = A * B
__global__ void matrixMultiplySharedForCov(double * A, double * B, double * C,int numARows, int numAColumns,int numBRows, int numBColumns,int numCRows, int numCColumns) 
{
    __shared__ double sA[32][32];   // Tile size of 32x32 
    __shared__ double sB[32][32];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    double Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ 32) + 1); k++)
    {
        if ( (Row < numARows) && (threadIdx.x + (k*32)) < numAColumns)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*32)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }            
        if ( Col < numBColumns && (threadIdx.y + k*32) < numBRows)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }            
        __syncthreads();

        for (int j = 0; j < 32; ++j)
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)
    {
        C[Row*numCColumns + Col] = Cvalue/(numARows-1);
    }
}



// Compute C = A * B
__global__ void matrixMultiplyShared(double * A, double * B, double * C,int numARows, int numAColumns,int numBRows, int numBColumns,int numCRows, int numCColumns) 
{
    __shared__ double sA[32][32];   // Tile size of 32x32 
    __shared__ double sB[32][32];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    double Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ 32) + 1); k++)
    {
        if ( (Row < numARows) && (threadIdx.x + (k*32)) < numAColumns)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*32)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }            
        if ( Col < numBColumns && (threadIdx.y + k*32) < numBRows)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }            
        __syncthreads();

        for (int j = 0; j < 32; ++j)
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)
    {
        C[Row*numCColumns + Col] = Cvalue;
    }
}




/*
A = Data
B = MCData.T
C = MCData
D = CovMatrx
*/


//dimensions perfect for rect matrix also
void flipMatriceCPU(Matrix A, double *matrice, int row, int col)
{
    int i, j;
    double tmp;
    for ( i = 0; i < row/2; i++ ) {
        for (  j = 0; j < col; j++ ) {
            tmp=matrice[col*i+j];
            A.elements[col*i+j] = matrice[col*(row-i-1)+j];
            A.elements[col*(row-i-1)+j] = tmp;

        }
    }
}


void printData(Matrix C,int height,int width)
{
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            cout<<C.elements[i*C.width+j]<<"  |   ";
        cout<<"--------------------------"<<endl;
    cout<<"\n\n============================"<<endl;
}


void getProjectedData(Matrix WTrans,Matrix mcData,Matrix proj)
{


    Matrix W;
    W.elements = (double*)malloc(sizeof(double)*WTrans.height*WTrans.width);
    W.width=WTrans.height;
    W.height=WTrans.width;


    //to store w
    Matrix d_W;
    d_W.width = d_W.stride = WTrans.height; 
    d_W.height = WTrans.width;
    size_t size = W.width * W.height * sizeof(double);
    cudaMalloc(&d_W.elements, size);


    //to store wtrans
    Matrix d_WTrans;
    d_WTrans.width = d_WTrans.stride = WTrans.width;
    d_WTrans.height = WTrans.height;
    size = WTrans.width * WTrans.height * sizeof(double);
    cudaMalloc(&d_WTrans.elements, size);
    cudaMemcpy(d_WTrans.elements, WTrans.elements, size,cudaMemcpyHostToDevice);

    dim3 dimGridN(WTrans.width/TILE_DIM, WTrans.height/TILE_DIM, 1);
    dim3 dimBlockN(TILE_DIM, BLOCK_ROWS, 1);

    //For calculating transpose of mean centered data
    MatTrans<<<dimGridN,dimBlockN>>>(d_W,d_WTrans);
    cudaMemcpy(W.elements, d_W.elements, size,cudaMemcpyDeviceToHost);

    //cout << "======WTrans running========"<<endl;
    //printData(WTrans,1,WTrans.width);

    //cout << W.elements[56]<<"#$#$#$";
    //cout <<"=======W running============="<<endl;
    //printData(W,W.height,1);


    dim3 dimGridMat(W.width/TILE_DIM, W.height/TILE_DIM, 1);
    dim3 dimBlockMat(TILE_DIM, TILE_DIM, 1);

    Matrix d_mcData;
    d_mcData.width = d_mcData.stride = mcData.width;
    d_mcData.height = mcData.height;
    size = mcData.width * mcData.height * sizeof(double);
    cudaMalloc(&d_mcData.elements, size);
    cudaMemcpy(d_mcData.elements, mcData.elements, size,cudaMemcpyHostToDevice);

    Matrix d_proj;
    d_proj.width = d_proj.stride = mcData.width;
    d_proj.height = mcData.height;
    size = mcData.width * mcData.height * sizeof(double);
    cudaMalloc(&d_proj.elements, size);
    cudaMemcpy(d_proj.elements, mcData.elements, size,cudaMemcpyHostToDevice);

    matrixMultiplyShared<<<dimGridMat,dimBlockMat>>>(d_mcData.elements,d_W.elements,d_proj.elements,d_mcData.height,d_mcData.width,d_W.height,d_W.width,d_proj.height,d_proj.width);
    cudaMemcpy(proj.elements, d_proj.elements, size,cudaMemcpyDeviceToHost);

    //printData(proj,proj.height,1);
}




//for rect matrix also dim compatabl;e, need to change block size
void MatMean(const Matrix A,Matrix B, Matrix C,Matrix D)
{
    // Load A and B to device memory
    //int N = 4096;//((A.width * A.height))/threadsPerBlock;

    Matrix d_A;
    d_A.width = d_A.stride = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,cudaMemcpyHostToDevice);



   
    //to store mean centered data
    Matrix d_C;
    d_C.width = d_C.stride = A.width; 
    d_C.height = A.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc(&d_C.elements, size);


    //for storing transpose of A so reverse dim of A
    Matrix d_B;
    d_B.width = d_B.stride = A.height;
    d_B.height = A.width;
    size = B.width * B.height * sizeof(double);
    cudaMalloc(&d_B.elements, size);


    //to store covariance matrix which is of features*features...
    Matrix d_D;
    d_D.width = d_D.stride = A.width; 
    d_D.height = A.width;
    size = D.width * D.height * sizeof(double);
    cudaMalloc(&d_D.elements, size);




    dim3 dimGrid(A.width/TILE_DIM, A.height/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);


    clock_t m2Start = clock();
    cout<<"making data mean centred..."<<endl;
    //For calculating mean centered data
    MatMeanKernel<<<dimGrid,dimBlock>>>(d_A,d_C,A.height,A.width);
    cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
    printf("time taken for making data mean centred: %.2fs\n", (double)(clock() - m2Start)/CLOCKS_PER_SEC);



    // cout<<endl<<"=======mean centered data======="<<endl;
    // for(int i=0;i<C.height;i++)
    //     for(int j=0;j<C.width;j++)
    //         cout<<C.elements[i*C.width+j]<<" $$$ ";
    //     cout<<"======xxxx========="<<endl;
    //     cout << "######" <<endl;
    // cout<<"\n\n============================"<<endl;

    
    //For calculating transpose of mean centered data

    clock_t covStart = clock();
    cout<<"making co-variance matrix..."<<endl;
    MatTrans<<<dimGrid,dimBlock>>>(d_B,d_C);
    cudaMemcpy(B.elements, d_B.elements, size,cudaMemcpyDeviceToHost);

    printf("time taken for making co-variance matrix: %.2fs\n", (double)(clock() - covStart)/CLOCKS_PER_SEC);

    // cout<<endl<<"=======Transposed mean centered data======="<<endl;
    // // for(int i=0;i<B.height;i++)
    // //     for(int j=0;j<1;j++)
    // //         cout<<B.elements[i*B.width+j]<<"   @@   ";
    // //     cout<<endl;
    // //     cout << "######"<<endl;
    // cout<<"\n\n============================"<<endl;



    // matrixMultiplyShared<<<dimGridM,dimBlockM>>>(double * A, double * B, double * C,int numARows, int numAColumns,int numBRows, int numBColumns,int numCRows, int numCColumns) 

    dim3 dimBlockm(TILE_DIM, TILE_DIM, 1);


    matrixMultiplySharedForCov<<<dimGrid,dimBlockm>>>(d_B.elements,d_C.elements,d_D.elements,d_B.height,d_B.width,d_C.height,d_C.width,d_D.height,d_D.width);
    cudaMemcpy(D.elements, d_D.elements, size,cudaMemcpyDeviceToHost);

    // cout<<endl<<"=======Multiplied A & A.T ======="<<endl;

    // cout<<D.elements[0]<<"   *****   ";

    // for(int i=0;i<D.height;i++)
    //     for(int j=0;j<D.width;j++)
    //         cout<<D.elements[i*D.width+j]<<"   *****   ";
    //     cout<<endl;
    //     cout << "######"<<endl;
    // cout<<"\n\n============================"<<endl;


    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
    cudaFree(d_B.elements);
    cudaFree(d_D.elements);
}


int main()
{
    int inp_height,inp_width;
    cout << "Enter Height:";
    cin >> inp_height;
    cout << "Enter Width:";
    cin >> inp_width;

    Matrix X,C,B,D,weiTrans,proj;

    X.width = inp_width;
    X.height = inp_height;


    B.width = inp_width;
    B.height = inp_height;


    C.width = inp_width;
    C.height = inp_height;
    

    D.width = inp_width;
    D.height = inp_height;


    weiTrans.width = inp_width;
    weiTrans.height = inp_height;


    proj.width = inp_width;
    proj.height = inp_height;


    double *p = (double*)malloc(sizeof(double)*inp_width*inp_height);
    double *q = (double*)malloc(sizeof(double)*inp_width*inp_height);

    double *r = (double*)malloc(sizeof(double)*inp_width*inp_height);
    double *s = (double*)malloc(sizeof(double)*inp_width*inp_height);

    double *t = (double*)malloc(sizeof(double)*inp_width*inp_height);
    double *u = (double*)malloc(sizeof(double)*inp_width*inp_height);

    D.elements = s;
    X.elements = p;
    C.elements = q;

    B.elements = r;
    weiTrans.elements = t;
    proj.elements = u;



    for (int i = 0; i < inp_width*inp_height; ++i)
    {
         X.elements[i] = i;//rand()%100+1;

    }
    
    MatMean(X,B,C,D); 


    clock_t eigStart = clock();
    cout<<"finding eigen values and vectors of co-variance matrix..."<<endl;
    //=====================================================================================================//


        cusolverDnHandle_t cusolverH = NULL;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cudaStat1 = cudaSuccess;
        cudaError_t cudaStat2 = cudaSuccess;
        cudaError_t cudaStat3 = cudaSuccess;


        const int m = D.width;
        const int lda = m;

        double *A = D.elements;
        double *V =(double*)malloc(sizeof(double)*lda*m) ; // eigenvectors

        double *W = (double*)malloc(sizeof(double)*m) ;; // eigenvalues

        double *d_A = NULL;
        double *d_W = NULL;
        int *devInfo = NULL;
        double *d_work = NULL;
        int  lwork = 0;

        int info_gpu = 0;

    // step 1: create cusolver/cublas handle
        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // step 2: copy A and B to device
        cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
        cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
        cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);

        cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat1);

    // step 3: query working space of syevd
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        cusolver_status = cusolverDnDsyevd_bufferSize(
            cusolverH,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_W,
            &lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
        assert(cudaSuccess == cudaStat1);

    // step 4: compute spectrum
        cusolver_status = cusolverDnDsyevd(
            cusolverH,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_W,
            d_work,
            lwork,
            devInfo);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);

        cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);


        // printf("================Eigenvalue ascending order==============\n");
        // for(int i = 0 ; i < m ; i++){
        //     printf("W[%d] = %E\n", i+1, W[i]);
        // }

        // printf("================Eigen Vectors===============\n");
        // //printMatrix(m, m, V, lda, "V");
        // printf("=====\n");


        if (d_A    ) cudaFree(d_A);
        if (d_W    ) cudaFree(d_W);
        if (devInfo) cudaFree(devInfo);
        if (d_work ) cudaFree(d_work);

        if (cusolverH) cusolverDnDestroy(cusolverH);

        cudaDeviceReset();

    //=========================================================================================//



    //to get w transpose from  cuSolver output
    flipMatriceCPU(weiTrans,V,inp_height,inp_width);

    printf("time taken for finding eigen values & eigen vectors: %.2fs\n", (double)(clock() - eigStart)/CLOCKS_PER_SEC);

    //printData(weiTrans,weiTrans.height,1);

    clock_t mm2Start = clock();
    cout<<"projecting data..."<<endl;

    //to get w from wT
    getProjectedData(weiTrans,C,proj);
    printf("time taken for projecting: %.2fs\n", (double)(clock() - mm2Start)/CLOCKS_PER_SEC);


    // printf("================PROJECTED DATA===============\n");


    // for (int i = 0; i < 16; ++i)
    // {
        
    //     cout<<proj.elements[i]<<endl;
        
    // }

    cout<<endl;
return 0;
}