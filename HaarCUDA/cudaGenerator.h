#include <stdio.h>

#pragma once

/*
библиотеку cvtypes.h подключить нельзя, т.к. в ней отсутствует поддержка С++
Ошибки:	ERROR C4430, ERROR C2144
*/

class cudaGenerator
{
	struct _gpu_rect
	{
		unsigned short p1;
		unsigned short p2;
		unsigned short p3;
		unsigned short p4;
	};

	struct _gpu_node
	{
		float a;
		float b;
		float threashold;
	};

	struct _gpu_cascade 
	{
		unsigned int node2_count;
		unsigned int node2_first;
		unsigned int node3_count;
		unsigned int node3_first;
		unsigned int node_position;
		
		float threashold;
	};


	_gpu_rect* __gpu_rect2s;
	_gpu_rect* __gpu_rect3s;
	float* __haar_rect2_weights;
	float* __haar_rect3_weights;
	_gpu_node* __haar_nodes;
	_gpu_cascade* __haar_cascade;

	#define _isq 1.0/(18.0*18.0)

	public:
		int GenerateCascade(void* _cascade, int cascadeStart, int cascadeStop, int cascadePerKernel, int threadsPerPixel,
					 bool **wasGen, FILE *_kernelFile, FILE *_JITFile, char ***, int);
		cudaGenerator(void) { };
		~cudaGenerator(void) { };
		void generateHeader(void *_kernelFile);
	private:
		int getCascades(void* _cascade_, int first_cascade, int cascadeCount, int gpu_window_width);
		void generateInitialization_FirstStage(FILE *fp);
		void generateFinalization_FirstStage(FILE *fp);
		void generateMiddle_FirstStage(int cascadeCount, FILE *fp);
		void generateInitialization(int index, int mode, int indexElement, FILE *fp);
		void generateFinalization(int mode, FILE *fp);
		void generateMiddle(int base_position, int mode, int cascadeCount, FILE *fp);
};
