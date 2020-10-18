#include "cudaAdditionFunctions.h"

#include <stdlib.h>
#include <cxtypes.h>
#include <cvtypes.h>
#include <svldpgm.h>

void cudaAdditionFunctions::_calcVarainceNormfactor(double *sqintegral_img, unsigned int *integral_img, unsigned int *_integral_img,
							 float *varince_result, int integral_img_height,
							 const int integral_img_width, const int integral_img_buffer_width, 
							 const int window_stepX, const int window_stepY,
							 const int windows_height, const int windows_width, const int windows_width32,
							 const int max_threads)
{
	const double tmp = 1.0/(18.0*18.0);

#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(_integral_img, integral_img)
#endif
	for (int i=0;i<integral_img_height;i++)
		for (int j=0;j<integral_img_width;j++)
			_integral_img[integral_img_buffer_width*i + j] = integral_img[integral_img_width*i + j];

#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(integral_img, sqintegral_img, varince_result)
#endif
	for (int i=1;i<windows_height;i++)
		for (int j=1;j<windows_width;j++)
		{
			double sqsum_img = sqintegral_img[i*integral_img_width + j] - sqintegral_img[(i)*integral_img_width + j + window_stepX]
				- sqintegral_img[(i+window_stepY)*integral_img_width + j] + sqintegral_img[(i+window_stepY)*integral_img_width + j + window_stepX]; 
			int sum = integral_img[i*integral_img_width + j] - integral_img[(i)*integral_img_width + j + window_stepX]
				- integral_img[(i+window_stepY)*integral_img_width + j] + integral_img[(i+window_stepY)*integral_img_width + j + window_stepX];
			double mean = sum*tmp;
			double variance_norm_factor = sqsum_img*tmp - mean*mean;
			if ( variance_norm_factor >= 0. )
				variance_norm_factor = sqrt(variance_norm_factor);
			else
				variance_norm_factor = 1.0;
			varince_result[windows_width32*(i-1)+(j-1)] = (float) variance_norm_factor;
		}
}

void cudaAdditionFunctions::RestoreMask32(unsigned char *res_mask, unsigned short *plans_cpu, unsigned char* res_cpu,
				 const int blockDimY32, const int blockDimX32,
				 const int windows_width, const int windows_height,
				 const int max_threads)
{
	memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));

#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(res_mask, plans_cpu)
#endif
	for (int i = 0; i < blockDimY32; i++ )
		for (int j = 0; j < blockDimX32; j++ )
			for (int k = 0; k < 32; k++)
				for (int l = 0; l < 32; l++)	
				{
					int base_offset = ( (i * blockDimX32 + j)<<10) + (k<<5) + l;
					int _x = (plans_cpu[base_offset])&0x1F;
					int _y = (plans_cpu[base_offset])>>5;
					if ( res_cpu[base_offset] > 0 && plans_cpu[base_offset] <= 1024)
					{
						if ( (i << 5) +_y < windows_height && (j<<5) + _x < windows_width)
							res_mask[i * (windows_width << 5) + (j << 5 ) + _y * windows_width + _x] =  255;
					}
				}
#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(res_mask)
#endif
	for (int i = 0; i < windows_height; i++)
		// чистим правый крайний столбец маски
		res_mask[windows_width * (i + 1) - 1] = 0;
}

void cudaAdditionFunctions::RestoreMask16(unsigned char *res_mask, unsigned char* res_cpu,
				 const int blockDimY32, const int blockDimX32,
				 const int windows_width, const int windows_height,
				 const int max_threads)
{
	for (int i = 0; i < blockDimY32; i++ )
		for (int j = 0; j < blockDimX32; j++ )
			for (int k = 0; k < 16; k++)
				for (int l = 0; l < 16; l++)	
				{
					int offset  = (k*16 + l);
					if (i*16+k < windows_height && j*16 + l < windows_width )
						res_mask[i * windows_width * 16 + j * 16 + k * windows_width + l] = res_cpu[256 * ( i * blockDimX32 + j ) + offset];
				}
#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(res_mask)
#endif
	for (int i = 0; i < windows_height; i++)
		// чистим правый крайний столбец маски
		res_mask[windows_width * (i + 1) - 1] = 0;
}

int cudaAdditionFunctions::node2_3_counter (const int cascade_count, void *_stages, int *node2, int *node3)
{
	CvHaarStageClassifier *stages = (CvHaarStageClassifier *)_stages;
	int node2_counter = 0;
	int node3_counter = 0;
	int _error = 0;
#ifdef _OPENMP
	#pragma omp parallel for shared(stages, _error) reduction(+: node2_counter, node3_counter)
#endif
	for (int i=0; i<cascade_count;i++)
	{
		CvHaarClassifier* classifier = stages[i].classifier;
		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier->count != 1)
				_error = 1;
			if (classifier[j].haar_feature->rect[2].weight==0)
				node2_counter++;
			else
				node3_counter++;
		}
	}
	if (_error > 0)
		return -1;
	(*node2) = node2_counter;
	(*node3) = node3_counter;
	return 0;
}

void cudaAdditionFunctions::filterCPU(unsigned int *_src, const int img_width, const int img_height, const int max_threads)
{
	unsigned char *src = (unsigned char *)_src;
#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(src)
#endif
	for (int i = 0; i < img_height; i++)
		for (int j = 1; j < img_width; j++)
			if (src[i * img_width + j] > 0 && src[i * img_width + j - 1] > 0 )
				src[i * img_width + j] = 0;
}

void cudaAdditionFunctions::savePGM(char *fileName, unsigned char *src, int width, int height)
{
	FILE *fp = fopen(fileName,"wb");
	if ( fp != 0)		
	{
		save_pgmt(fp, width, height, src);
		fclose( fp );
	}
}