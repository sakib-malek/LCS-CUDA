/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

const unsigned int BLOCK_SIZE = 512; 


__global__ void lcs(int A_grid, int B_grid, int *tab_h, char *A, char *B, int *top, int *left, int A_sz, int B_sz, int loop) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    int j = threadIdx.x + blockIdx.x*blockDim.x;
    int i = loop - j;
    int diag, a, b, m;

    /*
    ------------------------
    |  diag    |    a       |
    -------------------------
    |    b     | tab[i][j]  |  //we want to calculate tab[i][j]  
    ------------------------
    */

    //if (threadIdx.x == 0 ) {
			//do some memcpy to shared mem for optimization
	//}

    if (i < 0 || j < 0 || i>= BLOCK_SIZE || j>=BLOCK_SIZE || (A_grid*BLOCK_SIZE+i) >= A_sz || (B_grid*BLOCK_SIZE+j) >= B_sz) {
		return ;
	}
	//printf("from cuda %d %d\n",i,j);
    if (i == 0) {
        diag = top[j];
        a    = top[j+1];
        if (j == 0) {
            b = left[i+1];
        } else {
            b = tab_h[i*BLOCK_SIZE + j-1];
        }
    } else if (j == 0) {
        diag = left[i];
        b    = left[i+1];
        //i==0 handled above
        a    = tab_h[(i-1)*BLOCK_SIZE+ j];
    } else {
        diag = tab_h[(i-1)*BLOCK_SIZE+ j-1];
        a    = tab_h[(i-1)*BLOCK_SIZE+ j];
        b    = tab_h[i*BLOCK_SIZE+ j-1];
    }

	if (A[A_grid*BLOCK_SIZE+i] == B[B_grid*BLOCK_SIZE+j]) {
		tab_h[i*BLOCK_SIZE+ j] = diag +1;
	} else {
		if (a > b) {
			tab_h[i*BLOCK_SIZE+ j]= a;
		} else {
			tab_h[i*BLOCK_SIZE+ j]= b;
		}
	}
}


void lcsKernel(int A_grid, int B_grid, int *tab_h, char *A,  char *B, int *top, int *left, int A_sz, int B_sz, int loop)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE
    int n = (A_sz>B_sz)? A_sz: B_sz;
    dim3	DimGrid(1,1,1);//would need to change it to 1,1,1
    dim3	DimBlock(BLOCK_SIZE,1,1);
    lcs<<<DimGrid, DimBlock>>>(A_grid, B_grid, tab_h, A, B, top, left, A_sz, B_sz, loop);

}

