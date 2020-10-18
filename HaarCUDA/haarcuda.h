#ifndef __SAMPLE_H__
#define __SAMPLE_H__

#include "cvtypes.h"

void MemoryFree();
int cudaSynchronize(int cyclesOnGPU);
void restoreMask(unsigned char *res_mask, const int blockDimX32, const int blockDimY32,
				const int windows_width, const int windows_height, const int max_threads, unsigned char *res_cpu, unsigned short *plans_cpu);
void RunHaarClassifierCascadeJITversion( unsigned int *integral_img, double *sqintegral_img,int integral_img_width,
							   int integral_img_height, int max_threads, int pos,
							   int integralImg_length, int sqsum_length, int blockDimX, int blockDimY, int blockDimX32, int blockDimY32,
							   int integral_img_buffer_width, int windows_width32,
							   int window_stepX, int window_stepY, int windows_height, int windows_width, int blockDimY256_32,
							   int plans_length, int res_length,
							   unsigned char **_res_cpu, unsigned short **_plans_cpu);
void prepareCUDA( CvHaarClassifierCascade* cascade, int integral_img_width, int integral_img_height, int *profile, int length,
				  int *integralImg_lengthOut, int *sqsum_lengthOut, int *blockDimXOut, int *blockDimYOut, int *blockDimX32Out, int *blockDimY32Out,
				  int *integral_img_widthOut, int *integral_img_heightOut, int *integral_img_buffer_widthOut, int *windows_width32Out,
				  int *window_stepXOut, int *window_stepYOut, int *windows_heightOut, int *windows_widthOut, int *blockDimY256_32Out,
				  int *plans_lengthOut, int *res_lengthOut, int size);
#endif