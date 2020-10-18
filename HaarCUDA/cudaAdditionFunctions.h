#pragma once

class cudaAdditionFunctions
{
public:
	cudaAdditionFunctions(void) { };
	~cudaAdditionFunctions(void) { };
	void _calcVarainceNormfactor(double *sqintegral_img, unsigned int *integral_img, unsigned int *_integral_img,
							 float *varince_result, int integral_img_height,
							 const int integral_img_width, const int integral_img_buffer_width, 
							 const int window_stepX, const int window_stepY,
							 const int windows_height, const int windows_width, const int windows_width32,
							 const int max_threads);
	void RestoreMask32(unsigned char *res_mask, unsigned short *plans_cpu, unsigned char* res_cpu,
				 const int blockDimY32, const int blockDimX32,
				 const int windows_width, const int windows_height,
				 const int max_threads);
	void RestoreMask16(unsigned char *res_mask, unsigned char* res_cpu,
				 const int blockDimY32, const int blockDimX32,
				 const int windows_width, const int windows_height,
				 const int max_threads);
	int node2_3_counter (const int cascade_count, void *_stages, int *node2, int *node3);
	void filterCPU(unsigned int *_src, const int img_width, const int img_height, const int max_threads);
	void savePGM(char *fileName, unsigned char *src, int width, int height);
};
