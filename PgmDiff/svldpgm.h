
/* AUTHOR: Dmitry A. Bushenko
 * CREATED: 05.09.2006
 */

#ifndef SVLDPGM_H
#define SVLDPGM_H

#include <stdio.h>
 
int load_pgm(FILE* fp,int* width, int* height, unsigned char** data);
int save_pgm(FILE* fp,int width, int height, unsigned char* data);
int save_pgmt(FILE* fp,int width, int height, unsigned char* data);

int save_pgm4(FILE* fp,int width, int height, unsigned int* data);
int save_pgm4t(FILE* fp,int width, int height, unsigned int* data);
int load_pgm4(FILE* fp,int* width, int* height, unsigned int** data);

int save_pgm8f(FILE* fp,int width, int height, double* data);
int load_pgm8f(FILE* fp,int* width, int* height, double** data);

#endif
