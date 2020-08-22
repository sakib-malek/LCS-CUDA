/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

	char *A_h, *B_h;
	char *A_d, *B_d;
	int *tab_h, *tab_d, *table;
    int *top_d, *left_d;
    int *top_h, *left_h;	
    
    size_t A_sz, B_sz, diag_loop;
    int loop, i,j, k, l, m;
    int A_block_dim, B_block_dim;
    int A_grid, B_grid;

    
    unsigned VecSize;
   
    //dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 5120;
        A_sz = VecSize;
        B_sz = VecSize;
    } else if (argc == 2) {
    	VecSize = atoi(argv[1]);  
        if (VecSize > 40960) {
    		VecSize = 40960;
    	}
        A_sz = VecSize;
        B_sz = VecSize;
    } else if (argc == 3) {
        A_sz = atoi(argv[1]);  
        B_sz = atoi(argv[2]);
    } else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    printf("Dimention of:\nA %d\nB %d\n", A_sz, B_sz);

	
    //setting up input__________ start
    
    //set standard seed
    srand(217);
    
    //A_sz + 1 for null pointer
    A_h = (char*) malloc( sizeof(char)*(A_sz+1));
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%26)+'a';}
    
    //B_sz + 1 for null pointer
    B_h = (char*) malloc( sizeof(char)*(B_sz+1) );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%26)+'a';}
    //A_h[0] = B_h[0] = 'A';//adding a char out of range as null
    A_h[A_sz] = B_h[B_sz] = '\0';//null pointer at the end

    //printf("A %s\nB %s\n",A_h,B_h);

    //setting up input__________ finish



    top_h  = (int*) malloc (sizeof(int) * (BLOCK_SIZE+1));
    left_h = (int*) malloc (sizeof(int) * (BLOCK_SIZE+1));
	
	tab_h = (int*) malloc (sizeof(int) * (BLOCK_SIZE*BLOCK_SIZE));
    table = (int*) malloc (sizeof(int) * (A_sz*B_sz));
    if (table == NULL) {
        printf("table malloc failed\n");
    }
	

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMalloc((void**) &A_d, sizeof(char) * (A_sz+1));
    cudaMalloc((void**) &B_d, sizeof(char) * (B_sz+1));
    cudaMalloc((void**) &tab_d, sizeof(int) * (BLOCK_SIZE*BLOCK_SIZE));
    cudaMalloc((void**) &top_d, sizeof(int) * (BLOCK_SIZE+1));
    cudaMalloc((void**) &left_d, sizeof(int) * (BLOCK_SIZE+1));

    cudaMemset(tab_d, 0, sizeof(int) * (BLOCK_SIZE*BLOCK_SIZE));//always


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(A_d, A_h, sizeof(char) * (A_sz+1), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(char) * (B_sz+1), cudaMemcpyHostToDevice);

    //cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    //breaking up the input into tiles/grids
    A_block_dim = (A_sz-1)/BLOCK_SIZE +1;
    B_block_dim = (B_sz-1)/BLOCK_SIZE +1;

    for (A_grid = 0; A_grid < A_block_dim; A_grid++) {
        for (B_grid = 0; B_grid < B_block_dim; B_grid++) {
            if (A_grid == 0) {
                cudaMemset(top_d, 0, sizeof(int) * (BLOCK_SIZE+1));
            } else {
                if (B_grid == 0) {//corner case
                    top_h[0] = 0;
                } else {
                    top_h[0] = table[(A_grid*BLOCK_SIZE-1)*B_sz + B_grid*BLOCK_SIZE-1];
                }
                memcpy((top_h+1),(table + (A_grid*BLOCK_SIZE-1)*B_sz + B_grid*BLOCK_SIZE), sizeof(int) * BLOCK_SIZE);
                cudaMemcpy(top_d, top_h, sizeof(int) * (BLOCK_SIZE+1), cudaMemcpyHostToDevice);
            }

            if (B_grid == 0) {
                cudaMemset(left_d, 0, sizeof(int) * (BLOCK_SIZE+1));
            } else {
                if (A_grid == 0) {//corner case
                    left_h[0] = 0;
                } else {
                    left_h[0] = table[(A_grid*BLOCK_SIZE-1)*B_sz + B_grid*BLOCK_SIZE-1];
                }
                for (m = 1; m <= BLOCK_SIZE; m++) {
                    left_h[m] = table[(A_grid*BLOCK_SIZE+m-1)*B_sz + B_grid*BLOCK_SIZE-1];
                }
                cudaMemcpy(left_d, left_h, sizeof(int) * (BLOCK_SIZE+1), cudaMemcpyHostToDevice);
            }

            diag_loop = 2*BLOCK_SIZE;
            for (loop = 0;loop <= diag_loop; loop++) {
                lcsKernel(A_grid, B_grid, tab_d, A_d, B_d, top_d, left_d, A_sz, B_sz, loop);
            }

            cudaMemcpy(tab_h, tab_d, sizeof(int) * (BLOCK_SIZE*BLOCK_SIZE), cudaMemcpyDeviceToHost);

            for (k = 0; k < BLOCK_SIZE; k++) {
                for (l = 0; l < BLOCK_SIZE; l++) {
                    if ( ((A_grid*BLOCK_SIZE+k) < A_sz) && ((B_grid*BLOCK_SIZE+l) < B_sz) ) {
                        table[(A_grid*BLOCK_SIZE+k)*B_sz + (B_grid*BLOCK_SIZE+l)] = tab_h[k*BLOCK_SIZE+l];
                    }
                }
            }
        }
    }
    

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    //INSERT CODE HERE

    //cudaDeviceSynchronize();

    // Verify correctness -----------------------------------------------------
    verify(A_h, B_h, table, A_sz, B_sz);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(tab_h);
    free(top_h);
    free(left_h);
    free(table);

    //INSERT CODE HERE
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(tab_d);
    cudaFree(top_d);
    cudaFree(left_d);

    return 0;

}
