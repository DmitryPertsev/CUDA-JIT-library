#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

class cudaDriverMode
{
	CUdeviceptr *sqsum_img_gpuD;
	CUdeviceptr *integral_img_gpuD;
	CUdeviceptr *res_gpuD, *res_gpu_1cascadeD;
	CUdeviceptr *elements_gpuD;
	CUdeviceptr *plans_gpuD, *_plans_gpuD;

	unsigned short **plans_cpu;
	unsigned char **res_cpu;
	float **varince_result;
	unsigned int **_integral_img;

	int integralImageLength;
	int sqsumAllignedLength;

	int integral_img_width;
	int integral_img_height;
	int integral_img_buffer_width;
	int windows_width32;

	// используюся для корректной чистки памяти
	int cycles;
	int *profiles;
	int profileLength;

	CUcontext    hContext;
	CUdevice     hDevice;
	CUmodule	 cuModule;

	dim3 threads;
	dim3 threads256;
	dim3 blocks;
	dim3 blocks32;
	CUstream *stream;

	void *ptx;
	char ***functionName;

	void **firstStageParam;
	void **filterParam;
	void **dataToQueueB1Param;
	void **nextStage1_Param;
	void **dataToQueueB;
	void **nextStage2_Param;

	int *initStreams;

	FILE *fp;

	private:
		void* prepareToWork();
		void* getHaarFirstStageParams( int pos, int );
		void* getHaarNextStageParams( unsigned int *plansPointer, int pos, int index, int );
		void* getFilterParams(int pos, int index, int blockDimY256_32, int blockDimX32);
		void* getDataToQueueB1_32Params(int pos, int index);
		void* getDataToQueueBParams(int pos, int index);
		void cascadeGenerator(void* cascade, int *profile, int length);
		std::string findVisualStudioPath();
		std::string findCUDAToolkitPath();

	public:
		void calcVarainceNormfactor(int pos, double *sqintegral_img, unsigned int *integral_img, int integral_img_height,
							 const int integral_img_width, const int integral_img_buffer_width, 
							 const int window_stepX, const int window_stepY,
							 const int windows_height, const int windows_width, const int windows_width32,
							 const int max_threads);
		cudaDriverMode( int cyclesOnGPU, int integralImageLength, int resLength, int resAllignedLength,
						int sqsumLength, int sqsumAllignedLength, int plansLength, int elemLength,
						void *_cascade_, int *profile, int length/*, int pos*/);
		~cudaDriverMode(void);
		void setInputParameters(int , int , int, int, int, int, int , int , int , int );
		void execute(int pos, int windows_width, int windows_height, int blockDimY256_32,
					 int resLength, int planLength, int blockDimX32, int blockDimY32,
					 unsigned char **_res_cpu, unsigned short **_plans_cpu);
		int synchronize(int );
};
