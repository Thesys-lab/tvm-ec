/* C source code is found in dgemm_example.c */

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"



int main(int argc, char *argv[])
{
    if( argc != 5 ) {
      printf("Wrong number of arguments supplied (%d provided), please specify m, n, k, num_trials in the order.\n", argc);
      return 1;
    }

    float *A, *B, *C;
    int m, n, k, i, j, num_trials;
    float alpha, beta;

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    num_trials = atoi(argv[4]);

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function sgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are single precision scalars\n\n");

    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (float *)mkl_malloc( m*k*sizeof( float ), 32 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 32 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 32 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = (float)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (float)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    printf (" Computing matrix product using Intel(R) MKL sgemm function via CBLAS interface \n\n");
    // warm-up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    double start = dsecnd();
    for (int i = 0; i < num_trials; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    double end = dsecnd();
    printf ("\n Computations completed for %d trials.\n\n", num_trials);

    printf (" Top left corner of matrix A: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(k,6); j++) {
        printf ("%12.0f", A[j+i*k]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(k,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.0f", B[j+i*n]);
      }
      printf ("\n");
    }
    
    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C[j+i*n]);
      }
      printf ("\n");
    }

    printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf (" Example completed. \n\n");
    
    double duration = (end - start)/(double)num_trials;
    printf ("\nStart time: %f s\n", start);
    printf ("End time: %f s\n", end);
    printf ("Avg Duration: %f s\n\n", duration);
    double size = (double)m*k*sizeof( float ) + (double)k*n*sizeof( float ) + (double)m*n*sizeof( float );
    double bandwidth = size/duration/(1024*1024);
    printf ("Bandwidth: %f MB/s\n\n", bandwidth);
    return 0;
}