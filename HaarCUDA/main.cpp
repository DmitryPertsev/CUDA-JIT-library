#include <cxtypes.h>
#include <cvtypes.h>
#include "cudaDriverMode.h"

#ifdef DEBUG_MODE
#include "functions.h"
#endif

cudaDriverMode *_cuda = 0;

void MemoryFree()
{
	delete _cuda;
	printf("Memory free\n");
}

int iAlignUp(int a, int b)
{
	return (a % b != 0) ?  (a - a % b + b) : a;
}


void prepareCUDA( CvHaarClassifierCascade* cascade, int integral_img_width, int integral_img_height, int *profile, int length,
				  int *integralImg_lengthOut, int *sqsum_lengthOut, int *blockDimXOut, int *blockDimYOut, int *blockDimX32Out, int *blockDimY32Out,
				  int *integral_img_widthOut, int *integral_img_heightOut, int *integral_img_buffer_widthOut, int *windows_width32Out,
				  int *window_stepXOut, int *window_stepYOut, int *windows_heightOut, int *windows_widthOut, int *blockDimY256_32Out,
				  int *plans_lengthOut, int *res_lengthOut, int size)
{
	const int wsizew = 20, wsizeh = 20;
	int integral_img_width32 = iAlignUp(integral_img_width, 32);
	int integral_img_height32 = iAlignUp(integral_img_height, 32);
	
	int integral_img_buffer_width = integral_img_width32 + 32;
	int integral_img_buffer_height = integral_img_height32 + wsizeh;
	
	int windows_width =  integral_img_width-wsizew;
	int windows_height = integral_img_height-wsizeh;
	int windows_width32 = iAlignUp(windows_width, 32);
	int windows_height32 = iAlignUp(windows_height, 32);
	int windows_height256 = iAlignUp(windows_height, 256);
	int window_stepX = wsizew - 2;
	int window_stepY = wsizeh - 2;

	int blockDimX = (windows_width32>>4);
	int blockDimY = (windows_height32>>4);
	int blockDimY256_16 = (windows_height256>>4);

	int blockDimX32 = (windows_width32>>5);
	int blockDimY32 = (windows_height32>>5);
	int blockDimY256_32 = (windows_height256>>8);
	int blockDimY32_Alligned = iAlignUp(blockDimX32, 64);

	int plans_length 	= sizeof(unsigned short)	* blockDimX32		* (blockDimY32 << 10);
	int res_length 		= sizeof(unsigned char) 	* blockDimX 		* (blockDimY << 8);
	int res_length256	= sizeof(unsigned char) 	* blockDimX 		* (blockDimY256_16 << 8);
	int sqsum_length 	= sizeof(float) 			* windows_width32	* windows_height;
	int integralImg_length = integral_img_buffer_width * integral_img_buffer_height * sizeof(unsigned int);

	if (_cuda == 0)
		_cuda = new cudaDriverMode(size, integralImg_length, res_length, res_length256, sqsum_length,
								sizeof(float) * windows_width32 * windows_height32, plans_length,
								sizeof(unsigned short) * blockDimY32 * blockDimX32,
								cascade, profile, length);

	*integralImg_lengthOut = integralImg_length;
	*sqsum_lengthOut = sqsum_length;
	*blockDimXOut = blockDimX;
	*blockDimYOut = blockDimY;
	*blockDimX32Out = blockDimX32;
	*blockDimY32Out = blockDimY32;
	*integral_img_widthOut = integral_img_width;
	*integral_img_heightOut = integral_img_height;
	*integral_img_buffer_widthOut = integral_img_buffer_width;
	*windows_width32Out = windows_width32;
	*window_stepXOut = window_stepX;
	*window_stepYOut = window_stepY;
	*windows_heightOut = windows_height;
	*windows_widthOut = windows_width;
	*blockDimY256_32Out = blockDimY256_32;
	*plans_lengthOut = plans_length;
	*res_lengthOut = res_length;
}

void RunHaarClassifierCascadeJITversion( unsigned int *integral_img, double *sqintegral_img,int integral_img_width,
							   int integral_img_height, int max_threads, int pos,
							   int integralImg_length, int sqsum_length, int blockDimX, int blockDimY, int blockDimX32, int blockDimY32,
							   int integral_img_buffer_width, int windows_width32,
							   int window_stepX, int window_stepY, int windows_height, int windows_width, int blockDimY256_32,
							   int plans_length, int res_length,
							   unsigned char **_res_cpu, unsigned short **_plans_cpu)
{
	if (_cuda != 0)
	{
		_cuda->setInputParameters(integralImg_length, sqsum_length, blockDimX, blockDimY, blockDimX32, blockDimY32,
							 integral_img_width, integral_img_height, integral_img_buffer_width, windows_width32);
		_cuda->calcVarainceNormfactor(pos, sqintegral_img, integral_img, integral_img_height,
							 integral_img_width, integral_img_buffer_width, window_stepX, window_stepY,
							 windows_height, windows_width, windows_width32, max_threads);
		//_cuda->executeNew(pos, windows_width, windows_height, blockDimY256_32,
		//			 res_length, plans_length, blockDimX32, blockDimY32, _res_cpu, _plans_cpu);
		_cuda->execute(pos, windows_width, windows_height, blockDimY256_32,
					 res_length, plans_length, blockDimX32, blockDimY32, _res_cpu, _plans_cpu);
	}
}

int cudaSynchronize(int cyclesOnGPU)
{
	return _cuda->synchronize(cyclesOnGPU);
}

void restoreMask(unsigned char *res_mask, const int blockDimX32, const int blockDimY32,
				const int windows_width, const int windows_height, const int max_threads, unsigned char *res_cpu, unsigned short *plans_cpu)
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