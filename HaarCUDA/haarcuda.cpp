#include <cutil_inline.h>
#include <cxtypes.h>
#include <cvtypes.h>

#include "functions.h"
#include "Values.h"
//#include <cudaGenerator.h>
#include "JIT.h"
#include "_Kernels.cu"
//#include <JIT_Start.h>

#include "cudaDriverMode.h"
#include <kernels.cu>

#if __DEVICE_EMULATION__
	extern "C" bool InitCUDA(void)
	{
		return true;
	}
#else
	extern "C" bool InitCUDA(void)
	{
		int count = 0;
		int i = 0;

		cudaGetDeviceCount(&count);
		if(count == 0) {
			fprintf(stderr, "There is no device.\n");
			return false;
		}

		for(i = 0; i < count; i++) {
			cudaDeviceProp prop;
			if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
				if(prop.major >= 1) {
					break;
				}
			}
		}
		if(i == count) {
			fprintf(stderr, "There is no device supporting CUDA.\n");
			return false;
		}
		
		CUDA_SAFE_CALL(cudaSetDevice(i));
		
		printf("CUDA initialized.\n");
		return true;
	}
#endif
/*
extern "C" void testing(int *test, int *result)
{
	int *dev;
	
	cutilSafeCall(cudaMalloc((void**)&dev, 	sizeof(int) * 10000));
	cutilSafeCall(cudaMemcpy(dev, test, sizeof(int) * 10000, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(result, dev, sizeof(int) * 10000, cudaMemcpyDeviceToHost));
}
*/
void MemoryFree()
{
	cutilSafeCall(cudaFree(sqsum_img_gpu));
	cutilSafeCall(cudaFree(integral_img_gpu));
	cutilSafeCall(cudaFree(res_gpu));
	cutilSafeCall(cudaFree(plans_gpu));
	cutilSafeCall(cudaFree(_plans_gpu));

	cudaFreeHost(_integral_img);
	cudaFreeHost(varince_result);
	cudaFreeHost(res_cpu );
	cudaFreeHost(plans_cpu );
	
//	for (int i = 0; i < 15; i++)
//		cudaStreamDestroy(&cudaStream[i]);
	
	printf("Memory free\n");
}

// инициализирует внутреннее представление каскада
int SetCascadeToGPU(CvHaarClassifierCascade* cascade,int first_cascade, int cascade_count, int gpu_window_width1, int gpu_window_width2)
{
	if (cascade->count < first_cascade || cascade->count < first_cascade + cascade_count)
		return 0;
	CvHaarStageClassifier *stages = &cascade->stage_classifier[first_cascade];

	int node2_count = 0;
	int node3_count = 0;
	if (node2_3_counter (cascade_count, stages, &node2_count, &node3_count) < 0)
		return -1;
	if (cascade_count>NODES || (node2_count<<1)>RECTS2 || 3*node3_count>RECTS3 || node2_count + node3_count > NODES)
		return -1;

	gpu_rect* _gpu_rect2s = (gpu_rect*)malloc(sizeof(gpu_rect)*(node2_count<<1));
	gpu_rect* _gpu_rect3s = (gpu_rect*)malloc(sizeof(gpu_rect)*3*node3_count);
	float* _haar_rect2_weights = (float*)malloc(sizeof(float)*(node2_count<<1));
	float* _haar_rect3_weights = (float*)malloc(sizeof(float)*node3_count*3);

	gpu_node* _haar_nodes = (gpu_node*)malloc(sizeof(gpu_node)*(node2_count + node3_count));
	gpu_cascade * _haar_cascade = (gpu_cascade *)malloc(sizeof(gpu_cascade)*cascade_count);

	const float icv_stage_threshold_bias = 0.0001f;

	int node2_pos = 0, node3_pos = 0, node_pos = 0;
	for (int i=0; i<cascade_count; i++)
	{
		int node2_count = 0, node3_count = 0;
		int gpu_window_width = i > 1 ? gpu_window_width2 : gpu_window_width1;
		gpu_window_width = cascade->orig_window_size.width + gpu_window_width;  //16+20 || 32+20
		_haar_cascade[i].threashold = stages[i].threshold - icv_stage_threshold_bias;
		_haar_cascade[i].node2_first = node2_pos;
		_haar_cascade[i].node3_first = node3_pos;
		_haar_cascade[i].node_position = node_pos;
		CvHaarClassifier* classifier = stages[i].classifier;

		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier[j].haar_feature->rect[2].weight == 0)
			{
				CvRect *curr_rect = &classifier[j].haar_feature->rect[0].r;
				int x = curr_rect->x;
				int y = curr_rect->y;
				_gpu_rect2s[node2_pos].p1 = (y * gpu_window_width + x)<<2;
				_gpu_rect2s[node2_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				_gpu_rect2s[node2_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				_gpu_rect2s[node2_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				_haar_rect2_weights[node2_pos] = classifier[j].haar_feature->rect[0].weight * isq;

				curr_rect = &classifier[j].haar_feature->rect[1].r;
				node2_pos++;
				x = curr_rect->x;
				y = curr_rect->y;
				_gpu_rect2s[node2_pos].p1 = (y * gpu_window_width + x)<<2;
				_gpu_rect2s[node2_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				_gpu_rect2s[node2_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				_gpu_rect2s[node2_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				_haar_rect2_weights[node2_pos] = classifier[j].haar_feature->rect[1].weight * isq;
				node2_pos++;
				node2_count++;

				_haar_nodes[node_pos].threashold = classifier[j].threshold[0];
				_haar_nodes[node_pos].a = classifier[j].alpha[0];
				_haar_nodes[node_pos].b = classifier[j].alpha[1];
				node_pos++;
			}
			else
			{
				CvRect *curr_rect = &classifier[j].haar_feature->rect[0].r;
				int x = curr_rect->x;
				int y = curr_rect->y;
				_gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				_gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				_haar_rect3_weights[node3_pos] = classifier[j].haar_feature->rect[0].weight * isq;
				node3_pos++;
				curr_rect = &classifier[j].haar_feature->rect[1].r;
				x = curr_rect->x;
				y = curr_rect->y;
				_gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				_gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				_haar_rect3_weights[node3_pos] = classifier[j].haar_feature->rect[1].weight * isq;
				node3_pos++;  
				curr_rect = &classifier[j].haar_feature->rect[2].r;
				x = curr_rect->x;
				y = curr_rect->y;
				_gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				_gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				_gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				_haar_rect3_weights[node3_pos] = classifier[j].haar_feature->rect[2].weight * isq;
				node3_pos++;  
				node3_count++;
			}
		}

		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier[j].haar_feature->rect[2].weight != 0)
			{
				_haar_nodes[node_pos].threashold = classifier[j].threshold[0];
				_haar_nodes[node_pos].a = classifier[j].alpha[0];
				_haar_nodes[node_pos].b = classifier[j].alpha[1];
				node_pos++;
			}
		}

		_haar_cascade[i].node2_count = node2_count;
		_haar_cascade[i].node3_count = node3_count;
	}

	cutilSafeCall(cudaMemcpyToSymbol(haar_rects2, _gpu_rect2s, sizeof(gpu_rect)*(node2_count<<1)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rects3, _gpu_rect3s, sizeof(gpu_rect)*node3_count*3));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rect_weights2, _haar_rect2_weights, sizeof(float)*(node2_count<<1)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rect_weights3, _haar_rect3_weights, sizeof(float)*node3_count*3));
	cutilSafeCall(cudaMemcpyToSymbol(haar_nodes, _haar_nodes, sizeof(gpu_node)*(node2_count + node3_count)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_cascade, _haar_cascade, sizeof(gpu_cascade)*cascade_count));

	free(_gpu_rect2s);
	free(_gpu_rect3s);
	free (_haar_rect2_weights);
	free (_haar_rect3_weights);
	free (_haar_nodes);
	free (_haar_cascade);
	return 1;
}

/*void InitConstMemory( void *_gpu_rect2s, void *_gpu_rect3s,
					  float *_haar_rect2_weights, float *_haar_rect3_weights,
					  void *_haar_nodes, void *_haar_cascade,
					  int node2_count, int node3_count,  int cascade_count)
{
	cutilSafeCall(cudaMemcpyToSymbol(haar_rects2, _gpu_rect2s, sizeof(gpu_rect)*(node2_count<<1)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rects3, _gpu_rect3s, sizeof(gpu_rect)*node3_count*3));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rect_weights2, _haar_rect2_weights, sizeof(float)*(node2_count<<1)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_rect_weights3, _haar_rect3_weights, sizeof(float)*node3_count*3));
	cutilSafeCall(cudaMemcpyToSymbol(haar_nodes, _haar_nodes, sizeof(gpu_node)*(node2_count + node3_count)));
	cutilSafeCall(cudaMemcpyToSymbol(haar_cascade, _haar_cascade, sizeof(gpu_cascade)*cascade_count));

	free (_gpu_rect2s);
	free (_gpu_rect3s);
	free (_haar_rect2_weights);
	free (_haar_rect3_weights);
	free (_haar_nodes);
	free (_haar_cascade);
}*/

int iAlignUp(int a, int b)
{
	return (a % b != 0) ?  (a - a % b + b) : a;
}

//#define DEBUG_MODE

void RunHaarClassifierCascade(CvHaarClassifierCascade* cascade, int first_cascade, int cascade_count,  int gpu_window_width, 
							   unsigned int *integral_img, double *sqintegral_img,unsigned char* res_mask, int integral_img_width,
							   int integral_img_height, char* res_fname, int max_threads, bool isJIT, int pos,
							   int *_plansOffset, int *_resOffset, int *_blockDimX32, int *_blockDimY32,
							   int *_windows_width, int *_windows_height)
{
	const int wsizew = 20, wsizeh = 20;
	static bool initMemory = false;
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
	
#ifdef DEBUG_MODE
	unsigned int timer = 0;
	float elapsedTimeInMs = 0.0f, elapsedTimeInMs2 = 0.0f;
	cudaEvent_t start, stop;

	cutilCheckError( cutCreateTimer( &timer ) );
	cutilSafeCall  ( cudaEventCreate( &start ) );
	cutilSafeCall  ( cudaEventCreate( &stop ) );
#endif

	int blockDimX = (windows_width32>>4);
	int blockDimY = (windows_height32>>4);
	int blockDimY256_16 = (windows_height256>>4);

	int blockDimX32 = (windows_width32>>5);
	int blockDimY32 = (windows_height32>>5);
	int blockDimY256_32 = (windows_height256>>8);
	int blockDimY32_Alligned = iAlignUp(blockDimX32, 64);
	
	//mainStart();

	int plans_length 	= sizeof(unsigned short)	* blockDimX32		* (blockDimY32 << 10);
	int res_length 		= sizeof(unsigned char) 	* blockDimX 		* (blockDimY << 8);
	int res_length256	= sizeof(unsigned char) 	* blockDimX 		* (blockDimY256_16 << 8);
	int sqsum_length 	= sizeof(float) 			* windows_width32	* windows_height;
	int integralImg_length = integral_img_buffer_width * integral_img_buffer_height * sizeof(unsigned int);
	
	if (!initMemory)
	{
		for (int _i = 0; _i < MS; _i++)
			cudaStreamCreate(&cudaStream[_i]);

		cutilSafeCall(cudaMallocHost((void**)&_integral_img,	MS * integralImg_length));
		cutilSafeCall(cudaMallocHost((void**)&varince_result,	MS * sqsum_length));
		cutilSafeCall(cudaMallocHost((void**)&res_cpu,			MS * res_length));
		cutilSafeCall(cudaMallocHost((void**)&plans_cpu,		MS * plans_length));

		cutilSafeCall(cudaMalloc((void**)&sqsum_img_gpu,	MS * sizeof(float) * windows_width32 * windows_height32));
		cutilSafeCall(cudaMalloc((void**)&res_gpu,			MS * res_length256 ));
		cutilSafeCall(cudaMalloc((void**)&res_gpu_1cascade,	MS * res_length256 ));
		cutilSafeCall(cudaMalloc((void**)&integral_img_gpu,	MS * integralImg_length));
		cutilSafeCall(cudaMalloc((void**)&plans_gpu,		MS * plans_length));
		cutilSafeCall(cudaMalloc((void**)&_plans_gpu,		MS * plans_length));
		cutilSafeCall(cudaMalloc((void**)&elements_gpu,		MS * sizeof(unsigned short) * blockDimY32 * blockDimX32));
		
		memset( _integral_img,  0, MS * integralImg_length );
		memset( varince_result, 0, MS * sqsum_length );
		
		cutilSafeCall(cudaMemset(res_gpu,			0, MS * res_length256));	//
		cutilSafeCall(cudaMemset(res_gpu_1cascade,	0, MS * res_length256));
		cutilSafeCall(cudaMemset(plans_gpu,			0, MS * plans_length));
		cutilSafeCall(cudaMemset(_plans_gpu,		0, MS * plans_length));

		if (SetCascadeToGPU(cascade, 0, cascade_count, 16, 32) < 0)
		{
			printf("Couldn't initialize cascade\n");
			return;
		}
		
		initMemory = true;
		
		printf("Memory succsessfully init\n");
	}
	
	int integralImgOffset	 = pos * integralImg_length;
	int varianceResultOffset = pos * sqsum_length;
	int resGPU_Offset		 = pos * res_length256;
	int plansOffset			 = pos * plans_length;
	int elementsOffset		 = pos * sizeof(unsigned short) * blockDimY32 * blockDimX32;

	_calcVarainceNormfactor(sqintegral_img, integral_img, _integral_img + integralImgOffset, varince_result + varianceResultOffset, integral_img_height,
							 integral_img_width, integral_img_buffer_width, window_stepX, window_stepY,
							 windows_height, windows_width, windows_width32, max_threads);

#ifdef DEBUG_MODE
	printf("blockDimY %d blockDimX %d\n", blockDimY, blockDimX);
#endif

	unsigned int *integralImagePointer = integral_img_gpu + integralImgOffset;
	float *sqsumPointer = sqsum_img_gpu + varianceResultOffset;
	unsigned short *elementsPointer = elements_gpu + elementsOffset;
	unsigned int *resGPU_Pointer = (unsigned int *)(res_gpu + resGPU_Offset);
	unsigned int *plansPointer = (unsigned int *)(plans_gpu + plansOffset);
	unsigned int *_plansPointer = (unsigned int *)(_plans_gpu + plansOffset);
	unsigned int *resGPU1_Pointer = (unsigned int *)(res_gpu_1cascade + resGPU_Offset);
	
	cutilSafeCall(cudaMemcpyAsync(integralImagePointer,	_integral_img + integralImgOffset, integralImg_length,	cudaMemcpyHostToDevice, cudaStream[pos]));
	cutilSafeCall(cudaMemcpyAsync(sqsumPointer,	varince_result + varianceResultOffset,	  sqsum_length, cudaMemcpyHostToDevice, cudaStream[pos]));

	dim3 threads 	= dim3(16, 16);
	dim3 threads256 = dim3(256);
	dim3 blocks  	= dim3(blockDimX, blockDimY);
	dim3 blocks32 	= dim3(blockDimX32, blockDimY32);

	integral_img_width--;
	integral_img_height--;
	windows_width--;
	windows_height--;

#ifdef DEBUG_MODE
	cutilCheckError( cutStartTimer  ( timer ) );	 
	cutilSafeCall  ( cudaEventRecord( start, 0 ) );	
#endif

	unsigned int *plans_gpu_tmp;
	
	if (isJIT == false)
	{
		haar_first_stage<<<blocks, threads, 0, cudaStream[pos]>>> ( integralImagePointer, sqsumPointer, resGPU1_Pointer, 
												integral_img_width, integral_img_height, integral_img_buffer_width,
												windows_width32, 2);
		
#ifdef DEBUG_MODE
		cudaThreadSynchronize();
		cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU1_Pointer,  	res_length, cudaMemcpyDeviceToHost)); 
		memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
		RestoreMask16(res_mask, res_cpu, blockDimY, blockDimX, windows_width, windows_height, max_threads);
		
		char str[256];
		sprintf(str,"%s_gpu.1.pgm",res_fname);
		savePGM( str, res_mask, windows_width, windows_height );
#endif
												
		KERNELS::filter<<<dim3(blockDimY256_32, blockDimX32),threads256, 0, cudaStream[pos]>>>(resGPU1_Pointer, resGPU_Pointer, windows_width32);
		KERNELS::DataToQueueB1_32<<<blocks32, threads256, 0, cudaStream[pos]>>>  ( resGPU_Pointer, plansPointer, elementsPointer );
		
		//template <int threadsCount, int cascadeCount, int cycleInc, int whenSaveLogic, int logicOffset, int saveLogic>
		_haar_next_stage_<2, 1, 64, 64, 64, 128> <<<blocks32, threads, 0, cudaStream[pos]>>> (integralImagePointer, sqsumPointer, resGPU_Pointer, 
												integral_img_width, integral_img_height, integral_img_buffer_width,
												windows_width32, 2, plansPointer, elementsPointer );

#ifdef DEBUG_MODE
		cudaThreadSynchronize();
		cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer,  	res_length, cudaMemcpyDeviceToHost)); 
		cutilSafeCall( cudaMemcpy(plans_cpu, plansPointer, plans_length, cudaMemcpyDeviceToHost));
		memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
		RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
		
		sprintf(str,"%s_gpu.2.pgm", res_fname);
		savePGM( str, res_mask, windows_width, windows_height );
#endif

		for (int i = 3; i < 8; i++)
		{
			KERNELS::DataToQueueB<<<blocks32,threads256, 0, cudaStream[pos]>>>  (resGPU_Pointer, plansPointer, _plansPointer, elementsPointer );
			
			_haar_next_stage_<4, 1, 32, 127, 0x60, 192><<<blocks32, threads, 0, cudaStream[pos]>>> (integralImagePointer, sqsumPointer, resGPU_Pointer,
								integral_img_width, integral_img_height, integral_img_buffer_width,
								windows_width32, i, _plansPointer, elementsPointer );
			
#ifdef DEBUG_MODE
			cudaThreadSynchronize();
			cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer,  	res_length, cudaMemcpyDeviceToHost)); 
			cutilSafeCall( cudaMemcpy(plans_cpu, _plansPointer, plans_length, cudaMemcpyDeviceToHost)); 
			memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
			RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
			
			sprintf(str,"%s_gpu.%d.pgm", res_fname, i);
			savePGM( str, res_mask, windows_width, windows_height );
#endif

			swap();
		}

		for (int i = 8; i < cascade_count - 1; i++)
		{
			KERNELS::DataToQueueB<<<blocks32,threads256, 0, cudaStream[pos]>>>  (resGPU_Pointer, plansPointer, _plansPointer, elementsPointer );
			_haar_next_stage_<8, 1, 16, 127, 0x70, 224><<<blocks32, threads, 0, cudaStream[pos]>>> ( integralImagePointer, sqsumPointer, resGPU_Pointer, 
									integral_img_width, integral_img_height, integral_img_buffer_width,
									windows_width32, i, _plansPointer, elementsPointer );

#ifdef DEBUG_MODE
			cudaThreadSynchronize();
			cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer,  	res_length, cudaMemcpyDeviceToHost)); 
			cutilSafeCall( cudaMemcpy(plans_cpu, _plansPointer, plans_length, cudaMemcpyDeviceToHost));
			memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
			RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
			
			sprintf(str,"%s_gpu.%d.pgm", res_fname, i);
 			savePGM( str, res_mask, windows_width, windows_height );
#endif

			swap();
		}

		KERNELS::DataToQueueB<<<blocks32,threads256, 0, cudaStream[pos]>>>  (resGPU_Pointer, plansPointer, _plansPointer, elementsPointer );
		_haar_next_stage_<8, 1, 16, 127, 0x70, 224><<<blocks32, threads, 0, cudaStream[pos]>>> (integralImagePointer, sqsumPointer, resGPU_Pointer, 
								integral_img_width, integral_img_height, integral_img_buffer_width,
								windows_width32, cascade_count - 1, _plansPointer, elementsPointer );

#ifdef DEBUG_MODE
		cudaThreadSynchronize();
		cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer,res_length, cudaMemcpyDeviceToHost)); 
		cutilSafeCall( cudaMemcpy(plans_cpu, _plansPointer, plans_length, cudaMemcpyDeviceToHost)); 
		memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
		RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
		
		sprintf(str,"%s_gpu.final.pgm", res_fname);
		savePGM( str, res_mask, windows_width, windows_height );
#endif

	}
	else
	{
		static void (*funcStart)(unsigned int *img ,float* sqsum, unsigned int *res,
								   unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
								   unsigned int sqsum_buffer_width, unsigned int cascade_count, unsigned int *plan,
								   unsigned short *elementIndex);

		haar_first_stage_1<<<blocks, threads, 0, cudaStream[pos]>>> ( integralImagePointer, sqsumPointer, resGPU1_Pointer, 
												integral_img_width, integral_img_height, integral_img_buffer_width,
												windows_width32, 2);

#ifdef DEBUG_MODE
		cudaThreadSynchronize();
		cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU1_Pointer,  	res_length256, cudaMemcpyDeviceToHost)); 
		memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
		RestoreMask16(res_mask, res_cpu, blockDimY, blockDimX, windows_width, windows_height, max_threads);
		
		char str[256];
		sprintf(str,"%s_gpu.1.pgm",res_fname);
		savePGM( str, res_mask, windows_width, windows_height );
#endif

		KERNELS::filter<<<dim3(blockDimY256_32, blockDimX32),threads256, 0, cudaStream[pos]>>>(resGPU1_Pointer, resGPU_Pointer, windows_width32);
		KERNELS::DataToQueueB1_32<<<blocks32,threads256, 0, cudaStream[pos]>>>  (resGPU_Pointer, plansPointer, elementsPointer );

		funcStart = (void (*)(unsigned int *, float *, unsigned int *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *, unsigned short *))funcAddress[pos][0];
		funcStart<<<blocks32, threads, 0, cudaStream[pos]>>> ( integralImagePointer, sqsumPointer, resGPU_Pointer, 
								integral_img_width, integral_img_height, integral_img_buffer_width,
								windows_width32, 2, plansPointer, elementsPointer );

#ifdef DEBUG_MODE
		cudaThreadSynchronize();
		cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer, res_length256, cudaMemcpyDeviceToHost)); 
		cutilSafeCall( cudaMemcpy(plans_cpu, plansPointer,   plans_length, cudaMemcpyDeviceToHost));
		memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
		RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
		
		sprintf(str,"%s_gpu.2.pgm", res_fname);
		savePGM( str, res_mask, windows_width, windows_height );
#endif

		int start = 1;
		while (funcAddress[pos][start] != NULL)
		{
			KERNELS::DataToQueueB<<<blocks32,threads256, 0, cudaStream[pos]>>>  (resGPU_Pointer, plansPointer, _plansPointer, elementsPointer );
			funcStart = (void (*)(unsigned int *, float *, unsigned int *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int *, unsigned short *))funcAddress[pos][start];
			funcStart<<<blocks32, threads, 0, cudaStream[pos]>>> ( integralImagePointer, sqsumPointer, resGPU_Pointer, 
							integral_img_width, integral_img_height, integral_img_buffer_width,
							windows_width32, 0, _plansPointer, elementsPointer );

#ifdef DEBUG_MODE
			cudaThreadSynchronize();
			cutilSafeCall( cudaMemcpy(res_cpu, 	 resGPU_Pointer,  	res_length256, cudaMemcpyDeviceToHost)); 
			cutilSafeCall( cudaMemcpy(plans_cpu, _plansPointer, plans_length, cudaMemcpyDeviceToHost));
			memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
			RestoreMask32(res_mask, plans_cpu, res_cpu, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
				
			sprintf(str,"%s_gpu.%d.pgm", res_fname, start + 2);
			savePGM( str, res_mask, windows_width, windows_height );
#endif

			swap();
			start++;
		}
	}

#ifdef DEBUG_MODE
	cudaThreadSynchronize();
	cutilSafeCall( cudaEventRecord( stop, 0 ) );
	cutilSafeCall( cudaThreadSynchronize() );
	cutilCheckError( cutStopTimer( timer));
#endif

	cutilSafeCall( cudaMemcpyAsync(res_cpu + resGPU_Offset, resGPU_Pointer, 	res_length, cudaMemcpyDeviceToHost, cudaStream[pos])); 
	cutilSafeCall( cudaMemcpyAsync(plans_cpu + plansOffset, _plansPointer,  plans_length, cudaMemcpyDeviceToHost, cudaStream[pos])); 

#ifdef DEBUG_MODE
	cutilSafeCall( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
	elapsedTimeInMs2 = cutGetTimerValue( timer);
	printf ("Time 1 %f ms\n",elapsedTimeInMs );
	printf ("Time 2 %f ms\n",elapsedTimeInMs2 );
	cutilSafeCall( cudaEventDestroy(stop) );
	cutilSafeCall( cudaEventDestroy(start) );
	cutilCheckError( cutDeleteTimer( timer));
#endif

	*_plansOffset	 = plansOffset;
	*_resOffset		 = resGPU_Offset;
	*_blockDimX32	 = blockDimX32;
	*_blockDimY32	 = blockDimY32;
	*_windows_width	 = windows_width;
	*_windows_height = windows_height;
}

cudaDriverMode *_cuda;

void RunHaarClassifierCascadeJITversion(CvHaarClassifierCascade* cascade, int first_cascade, int cascade_count,  int gpu_window_width, 
							   unsigned int *integral_img, double *sqintegral_img,unsigned char* res_mask, int integral_img_width,
							   int integral_img_height, char* res_fname, int max_threads, bool isJIT, int pos,
							   int *_plansOffset, int *_resOffset, int *_blockDimX32, int *_blockDimY32,
							   int *_windows_width, int *_windows_height, int *profile, int length,
							   unsigned char **_res_cpu, unsigned short **_plans_cpu)
{
	const int wsizew = 20, wsizeh = 20;
	static bool wasInit = false;
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

//	static CUcontext    hContext = 0;
//	static CUdevice     hDevice  = 0;

	int plans_length 	= sizeof(unsigned short)	* blockDimX32		* (blockDimY32 << 10);
	int res_length 		= sizeof(unsigned char) 	* blockDimX 		* (blockDimY << 8);
	int res_length256	= sizeof(unsigned char) 	* blockDimX 		* (blockDimY256_16 << 8);
	int sqsum_length 	= sizeof(float) 			* windows_width32	* windows_height;
	int integralImg_length = integral_img_buffer_width * integral_img_buffer_height * sizeof(unsigned int);

	if (!wasInit)
	{
		_cuda = new cudaDriverMode(MS, integralImg_length, res_length, res_length256, sqsum_length,
								sizeof(float) * windows_width32 * windows_height32, plans_length,
								sizeof(unsigned short) * blockDimY32 * blockDimX32,
								cascade, profile, length, pos);
		wasInit = true;
	}
	_cuda->setInputParameters(integralImg_length, sqsum_length, blockDimX, blockDimY, blockDimX32, blockDimY32,
							 integral_img_width, integral_img_height, integral_img_buffer_width, windows_width32);
	_cuda->calcVarainceNormfactor(pos, sqintegral_img, integral_img, integral_img_height,
							 integral_img_width, integral_img_buffer_width, window_stepX, window_stepY,
							 windows_height, windows_width, windows_width32, max_threads);
	_cuda->execute(pos, windows_width, windows_height, blockDimY256_32,
					 _blockDimX32, _blockDimY32, _windows_width, _windows_height,
					 res_length, plans_length, blockDimX32, blockDimY32, _res_cpu, _plans_cpu);
}

int cudaSynchronize(int cyclesOnGPU)
{
	//cudaThreadSynchronize();
	return _cuda->synchronize(cyclesOnGPU);
}

void restoreMask(unsigned char *res_mask/*, int plansOffset, int resOffset*/, int blockDimX32, int blockDimY32,
				int windows_width, int windows_height, int max_threads, unsigned char *_res_cpu, unsigned short *_plans_cpu)
{
	RestoreMask32(res_mask, _plans_cpu/* + plansOffset*/, _res_cpu/* + resOffset*/, blockDimY32, blockDimX32, windows_width, windows_height, max_threads);
}