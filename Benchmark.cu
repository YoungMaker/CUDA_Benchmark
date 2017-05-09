#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#define NUM_THREADS_PER_BLK 128
//# define ONE_THREAD_PER_BLOCK

void barf(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	exit(1);
}

unsigned long utime(void)
{
    struct timeval tv;
    unsigned long result = 0;

    gettimeofday(&tv, NULL);
    result += (tv.tv_sec * 1000000);
    result += tv.tv_usec;

    return result;
}


__host__ __device__ int getIndex(int rows, int cols, int row, int col) {
    return row * cols + col;
}


__global__ void cuMatrix(float *matrix_a, float *matrix_b, float *out, int rows_a, int cols_a, int rows_b, int cols_b) {

#ifdef ONE_THREAD_PER_BLOCK
		int i = blockIdx.x;
		int j = blockIdx.y;
#else
    int i = blockIdx.x * NUM_THREADS_PER_BLK + threadIdx.x;
    int j = blockIdx.y;

    if (i >= cols_b) {
        return;
    }
#endif

    //int i = blockIdx.y * blockDim.y + threadIdx.y ;   // Row i of matrix C
    //int j = blockIdx.x ;   // Column j of matrix C

    if(i*j > rows_a*cols_b ){return;}
		
    int out_index = i * cols_b + j;
	int a_index, b_index;
	float accum = 0;


    accum = 0;
    for(int k = 0; k < rows_b; k++){
        a_index = getIndex(rows_a, cols_a, i, k);
        b_index = getIndex(rows_b, cols_b, k, j);
        accum+= matrix_a[a_index] * matrix_b[b_index];
    }
	
	  out[out_index] = accum;
}


void allocMatrix(float *ptr, int rows, int cols) {
    for(int i = 0; i < cols*rows; i++) {
        ptr[i] = (float)drand48();
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    printf("[");
    for(int i=0; i < rows; i++) {
        for(int j = 0; j < cols; j++){
            if(j == (cols-1) && i == (rows-1)) {
                printf("%0.4f ]\n", matrix[getIndex(rows, cols, i, j)]);
                return;
            }
            printf("%0.4f ", matrix[getIndex(rows, cols, i, j)]);

        }
        printf("\n ");
    }
}

long gpu_bmark(int N, int M)
{
    srand48(time(0));

    int rows_a = N;
    int cols_a = M;

    int rows_b = cols_a; //rows of b must equal cols of a
    int cols_b = rows_a;

    //create and allocate CPU matrices.
    float *matrix_a = (float*)malloc(rows_a*cols_a*sizeof(float));
    float *matrix_b = (float*)malloc(rows_b*cols_b*sizeof(float));
    float *matrix_res = (float*)malloc(rows_a*cols_b*sizeof(float));

    allocMatrix(matrix_a, rows_a, cols_a);
    allocMatrix(matrix_b, rows_b, cols_b);

    //printMatrix(matrix_a, rows_a, cols_a);
    //printf("\n");
    //printMatrix(matrix_b, rows_b, cols_b);
    //printf("\n");

	for(int i=0; i<(rows_a*cols_b); i++) {
	   matrix_res[i] = 0;
	}

	float *dev_matrix_a, *dev_matrix_b, *dev_matrix_out;
	cudaMalloc( (void **) &dev_matrix_a, sizeof(float)*(rows_a)*(cols_a) );
	cudaMalloc( (void **) &dev_matrix_b, sizeof(float)*(rows_b)*(cols_b) );
	cudaMalloc( (void **) &dev_matrix_out, sizeof(float)*(rows_a)*(cols_b));

	// copy matrix data to device
	cudaMemcpy( dev_matrix_a, matrix_a, sizeof(float)*(rows_a)*(cols_a), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matrix_b, matrix_b, sizeof(float)*(rows_b)*(cols_b), cudaMemcpyHostToDevice );



    // compute grid dimensions
#ifdef ONE_THREAD_PER_BLOCK
    int num_blocks_x = cols_b;
    int num_threads = 1;
#else
	int num_blocks_x = cols_b / NUM_THREADS_PER_BLK;

	if ( cols_b % NUM_THREADS_PER_BLK != 0) {
		num_blocks_x++;
	}
    int num_threads = NUM_THREADS_PER_BLK;
#endif
	//dim3 grid(num_blocks_x, 1);

    dim3 grid(num_blocks_x, rows_a);

	unsigned long begin = utime();

	cuMatrix<<<grid, num_threads>>>( dev_matrix_a, dev_matrix_b, dev_matrix_out, rows_a, cols_a, rows_b, cols_b);

    // copy transformed matrix data from device
	cudaMemcpy( matrix_res, dev_matrix_out, sizeof(float)*(rows_a)*(cols_b), cudaMemcpyDeviceToHost );

   //printMatrix(matrix_res, rows_a, cols_b);

   cudaFree(dev_matrix_a);
   cudaFree(dev_matrix_b);
   cudaFree(dev_matrix_out);

   unsigned long end = utime();

   unsigned long elapsed = end - begin;
  // printf("computation took %lu microseconds\n", elapsed);

   free(matrix_a);
   free(matrix_b);
   free(matrix_res);
	return elapsed;
}


long cpu_bmark(int N, int M)
{

    int rows_a = N;
    int cols_a = M;

    int rows_b = cols_a;
    int cols_b = rows_a;

    float *matrix_a = (float*)malloc(rows_a*cols_a*sizeof(float));
    float *matrix_b = (float*)malloc(rows_b*cols_b*sizeof(float));
    float *matrix_res = (float*)malloc(rows_a*cols_b*sizeof(float));

    allocMatrix(matrix_a, rows_a, cols_a);
    allocMatrix(matrix_b, rows_b, cols_b);

    //printMatrix(matrix_a, rows_a, cols_a);
    // printf("\n");
    // printMatrix(matrix_b, rows_b, cols_b);
    // printf("\n");


    unsigned long begin = utime();

    int res_index, a_index, b_index;
    for(int i=0; i < rows_a; i++){
        for(int j=0; j < cols_b; j++){
            res_index = getIndex(rows_a, cols_b, i, j);
            matrix_res[res_index] = 0;
            for(int k=0; k < rows_b; k++){
                a_index = getIndex(rows_a,cols_a, i, k);
                b_index = getIndex(rows_b, cols_b, k, j);
                matrix_res[res_index] += matrix_a[a_index] * matrix_b[b_index];
            }
        }
    }
    //printMatrix(matrix_res, rows_a, cols_b);


    unsigned long end = utime();

    unsigned long elapsed = end - begin;

    free(matrix_a);
    free(matrix_b);
    free(matrix_res);
    return elapsed;
}

int main(int argc, char **argv) {
    printf("input size, GPU time\n");
    for(int i=5000; i<=8000; i+=250) {
        printf("%i, %li\n",i, gpu_bmark(i,i));
    }
}
