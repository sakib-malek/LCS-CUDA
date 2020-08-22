
/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

void lcs(char *A, char *B, int *tab, unsigned int A_sz, unsigned int B_sz) {
	int i,j;
	int diag, a, b;

	memset(tab, 0, sizeof(tab));

	/*
    ------------------------
    |  diag    |    a       |
    -------------------------
    |    b     | tab[i][j]  |  //we want to calculate tab[i][j]  
    ------------------------
    */
	
	for (i=0; i<A_sz;i++) {
		for (j=0;j<B_sz;j++) {

			if (i == 0) {
			    diag = 0;
			    a    = 0;
			    if (j == 0) {
			        b = 0;
			    } else {
			        b = tab[j-1];
			    }
			} else if (j == 0) {
			    diag = 0;
			    b    = 0;
			    //i==0 handled above
			    a    = tab[(i-1)*(B_sz)+ j];
			} else {
			    diag = tab[(i-1)*(B_sz)+ j-1];
			    a    = tab[(i-1)*(B_sz)+ j];
			    b    = tab[i*(B_sz)+ j-1];
			}

			if (A[i] == B[j]) {
				tab[i*B_sz+j] = diag +1;
			}
			else {
				tab[i*B_sz+j] = MAX(a,b);
			}
		}
	}
	return;
}


void verify(char *A, char *B, int *C, unsigned int A_sz, unsigned int B_sz) {
	int i,j;
	//pres is a the locally calculated value

	int *pres = (int*) malloc (sizeof(int) * ((A_sz)*(B_sz)));
	printf("Verifying results...\n"); fflush(stdout);
	
	//doing serial computations
	startTime(&timer);
	lcs(A,B, pres, A_sz, B_sz);
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	int flag = 1;
	for (i=0;i<A_sz;i++) {
		for (j=0;j<B_sz;j++) {
			if (C[i*B_sz+j] != pres[i*B_sz+j]) {
				flag =0;
			}
		}
	}
	if (flag) {
		printf("\nTEST PASSED\n\n");
	}

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
