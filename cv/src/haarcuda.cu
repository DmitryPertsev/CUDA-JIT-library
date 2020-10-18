/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cxtypes.h>
#include <cvtypes.h>
#include <svldpgm.h>

/************************************************************************/
/* Init CUDA															*/
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
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
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif

struct gpu_rect
{
	unsigned short p1;
	unsigned short p2;
	unsigned short p3;
	unsigned short p4;
};

struct gpu_node
{
	float a;
	float b;
	float threashold;
};

struct gpu_cascade 
{
	unsigned int node2_count;
	unsigned int node2_first;
	unsigned int node3_count;
	unsigned int node3_first;
	
	float threashold;
};

#define RECTS2		250		// 200		БЫЛО
#define RECTS3		120		// 120
#define NODES		371		// 321
#define CASCADES	2		// 2

__device__ __constant__ gpu_rect haar_rects2[RECTS2];
__device__ __constant__ float haar_rect_weights2[RECTS2];
__device__ __constant__ gpu_rect haar_rects3[RECTS3];
__device__ __constant__ float haar_rect_weights3[RECTS3];

__device__ __constant__ gpu_node  haar_nodes[NODES];
__device__ __constant__ gpu_cascade  haar_cascade[CASCADES];

#define window_STEPX 18
#define window_STEPY 18
#define isq 1.0/(18.0*18.0)

//#define DEBUG_MODE

/*
struct NODE2_TEST
{
	float threashold;
	float t;
	int sum1, sum2;
	float r1, r2;
	float rect_weights1, rect_weights2;
	float stage_sum;
	float a;
	float b;
};

struct NODE3_TEST
{
	float threashold;
	float t;
	int sum1, sum2, sum3;
	float r1, r2, r3;
	float rect_weights1, rect_weights2, rect_weights3;
	float stage_sum;
	float a;
	float b;
};

#define NODE2_COUNT 20
#define NODE3_COUNT 10
#define CASCADE_COUNT 2

struct CASCADE_TEST
{
	NODE2_TEST node2[NODE2_COUNT];
	NODE3_TEST node3[NODE3_COUNT];
	int node2_count;
	int node3_count;
	float variance_norm_factor;
	float stage_sum;
	float stage_sum_after_node2[2];
	float threashold;
};

struct DEBUG_INFO
{
	CASCADE_TEST test[CASCADE_COUNT];
	int cascade_count;
};
*/
// классификатор станартный
__global__ static void haar_first_stage(unsigned int *img ,float* sqsum, unsigned int *res, 
										unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
										unsigned int sqsum_buffer_width, unsigned int cascade_count)
{
	__shared__ unsigned int _img[36][36]; //(20+16) * (20+16)
	__shared__ float _sqsum[16][16];
	__shared__ unsigned char _res[256];
	int _imgX = threadIdx.x;
	int _imgY = threadIdx.y;
	
	unsigned int PosX = (blockIdx.x<<4) + threadIdx.x;
	unsigned int PosY = (blockIdx.y<<4) + threadIdx.y;
	
	int img_pos = __umul24(PosY, img_buffer_width) + PosX;
	
	_img[_imgY][_imgX] = img[img_pos];
	_img[_imgY][_imgX+16] = img[img_pos+16];
	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];
	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];
	if (threadIdx.x<4 && PosX+32<img_buffer_width )
	{
		_img[_imgY][_imgX+32] = img[img_pos + 32];
		_img[_imgY+16][_imgX+32] = img[img_pos+32 + (img_buffer_width<<4)];
	}
	if (threadIdx.y<4 && PosY+32<img_height)
	{
		_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];
		_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5)+16];
	}
	if (threadIdx.x<4 && threadIdx.y<4 && PosX+32<img_buffer_width && PosY+32<img_height )
	{
		_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5)+32];
	}

	int sqsum_pos = __umul24(PosY, sqsum_buffer_width) + PosX;
	_sqsum[_imgY][_imgX] = sqsum[sqsum_pos];
	int res_pos = (threadIdx.y<<4) + threadIdx.x;
	_res[res_pos] = 255;
	__syncthreads();
	
	if ((PosX + window_STEPX + 3<img_width) && (PosY + window_STEPY + 3<img_height))
	{
		unsigned char* cur_img = (unsigned char*)&_img[threadIdx.y][threadIdx.x]; 
		float variance_norm_factor = _sqsum[_imgY][_imgX];
		gpu_cascade* curr_cascade = haar_cascade;
		gpu_node* curr_node = &haar_nodes[0];
		
		for(int k=0;k<cascade_count && _res[res_pos] == 255 ;k++)
		{
			float stage_sum = 0.0f;
			gpu_rect *curr_rec = &haar_rects2[curr_cascade[k].node2_first];
			int j = curr_cascade[k].node2_first;
			for (int i=0; i<curr_cascade[k].node2_count;i++ )
			{
				float t = curr_node->threashold * variance_norm_factor;
				int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
				float r = (sum*haar_rect_weights2[j]);
				curr_rec++;
				j++;
				sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
				r += (sum*haar_rect_weights2[j]);
				stage_sum += r < t ? curr_node->a : curr_node->b;
				curr_rec++;
				j++;
				curr_node++;
			}

			curr_rec = &haar_rects3[curr_cascade[k].node3_first];
			j = curr_cascade[k].node3_first;
			for (int i=0; i<curr_cascade[k].node3_count;i++ )
			{
				float t = curr_node->threashold * variance_norm_factor;
				int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
				float r = (sum*haar_rect_weights3[j]);
				curr_rec++;
				j++;
				sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
				r += (sum*haar_rect_weights3[j]);
				curr_rec++;
				j++;
				sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
				r += (sum*haar_rect_weights3[j]);
				stage_sum += r < t ? curr_node->a : curr_node->b;
				curr_rec++;
				j++;
				curr_node++;
			}

			if( stage_sum < curr_cascade[k].threashold )	
				{
					_res[res_pos] = 0;
					break;
				}
			else 
				_res[res_pos] = 255; 
		}
	}
	__syncthreads();
	
	unsigned int *_resI = (unsigned int *)_res;
	if (threadIdx.y<4)
	{
		int resPos =  ( (__umul24(gridDim.x, blockIdx.y) + blockIdx.x)<<6 ) + res_pos;
		res[resPos] = _resI[res_pos];
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NODE2_CYCLE( i_add, curr_rec_add, j_add, curr_node_add ) {		\
	for (int i = arrayPos; i<curr_cascade[0].node2_count; i+=i_add )	\
	{																\
		float t = curr_node->threashold * variance_norm_factor;		\
		int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);	\
		float r = sum*haar_rect_weights2[j];						\
		curr_rec++;													\
		j++;														\
		sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);	\
		r += (sum*haar_rect_weights2[j]);							\
		stage_sum += r < t ? curr_node->a : curr_node->b;			\
		curr_rec += curr_rec_add;									\
		j += j_add;													\
		curr_node+=curr_node_add;									\
	}																\
}																	\

#define NODE3_CYCLE( i_add, curr_rec_add, j_add, curr_node_add ) {	\
	for (int i=arrayPos; i<curr_cascade[0].node3_count;i+=i_add )	\
	{																\
		float t = curr_node->threashold * variance_norm_factor;		\
		int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);	\
		float r = sum*haar_rect_weights3[j];						\
		curr_rec++;													\
		j++;														\
		sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);	\
		r += (sum*haar_rect_weights3[j]);							\
		curr_rec++;													\
		j++;														\
		sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);	\
		r += (sum*haar_rect_weights3[j]);							\
		stage_sum += r < t ? curr_node->a : curr_node->b;			\
		curr_rec+=curr_rec_add;										\
		j+=j_add;													\
		curr_node+=curr_node_add;									\
	}																\
}																	\

#define PLAN_ITEM(cycleInc, node2Inc, node3Inc, savePos) {									\
	if (plan_item <= 1024)																	\
	{																						\
		unsigned char* cur_img = (unsigned char*)&_img[(plan_item>>5)][plan_item&0x1F];		\
		float variance_norm_factor = _sqsum[plan_item];										\
		gpu_node* curr_node = &haar_nodes[arrayPos];										\
		gpu_rect *curr_rec = &haar_rects2[curr_cascade[0].node2_first] + (arrayPos<<1);		\
		int j = curr_cascade[0].node2_first + (arrayPos<<1);								\
		NODE2_CYCLE( cycleInc, node2Inc, node2Inc, cycleInc );								\
																							\
		curr_node = &haar_nodes[(arrayPos)+curr_cascade[0].node2_count];					\
		curr_rec = &haar_rects3[curr_cascade[0].node3_first] + __umul24(arrayPos, 3);		\
		j = curr_cascade[0].node3_first + __umul24(arrayPos, 3);							\
		NODE3_CYCLE( cycleInc, node3Inc, node3Inc, cycleInc );								\
																							\
		if ( ( arrayPos & 1 ) == 1)															\
			_stage_sum [savePos] = stage_sum;												\
	}																						\
	__syncthreads();																		\
}																							\
/*
#define PLAN_ITEM_BRANCH_2threads(base_offset) {											\
	PLAN_ITEM(2, 3, 4, plan_pos);															\
	if (arrayPos == 0 && plan_item <= 1024 )												\
	{																						\
		stage_sum = _stage_sum[plan_pos] + stage_sum;										\
		if( stage_sum < curr_cascade[0].threashold )										\
			_res[base_offset + plan_pos] = 0;												\
		else 																				\
			_res[base_offset + plan_pos] = 255; 											\
	}																						\
}																							\
*/
__global__ static void haar_next_stage(unsigned int *img ,float* sqsum, unsigned int *res, 
									   unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
									   unsigned int sqsum_buffer_width, unsigned int cascade_count, unsigned int *plan)
{
	__shared__ unsigned int _img[52][52]; //(20+16) * (20+16)
	__shared__ float _sqsum[32*32];
	__shared__ unsigned char _res[256];
	__shared__ unsigned short _plan[256];
	__shared__ float _stage_sum[128];
	int _imgX = threadIdx.x;
	int _imgY = threadIdx.y;
	
	unsigned int PosX = (blockIdx.x<<5) + threadIdx.x;
	unsigned int PosY = (blockIdx.y<<5) + threadIdx.y;
	
	int img_pos = __umul24(PosY, img_buffer_width) + PosX;
	
	_img[_imgY][_imgX] = img[img_pos];
	_img[_imgY][_imgX+16] = img[img_pos+16];
	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];
	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];
	
	_img[_imgY][_imgX+32] = img[img_pos+32];
	_img[_imgY+16][_imgX+32] = img[img_pos+32+(img_buffer_width<<4)];
	_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];
	_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5) + 16];
	_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5) + 32];
	
	if (threadIdx.x<4 && PosX+48<img_buffer_width )
	{
		_img[_imgY][_imgX+48] = img[img_pos + 48];
		_img[_imgY+16][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<4)];
		_img[_imgY+32][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<5)];
	}
	if (threadIdx.y<4 && PosY+48<img_height)
	{
		_img[_imgY+48][_imgX] = img[img_pos+__umul24(img_buffer_width, 48)];
		_img[_imgY+48][_imgX+16] = img[img_pos+__umul24(img_buffer_width, 48)+16];
		_img[_imgY+48][_imgX+32] = img[img_pos+__umul24(img_buffer_width, 48)+32];
	}
	if (threadIdx.x<4 && threadIdx.y<4 && PosX+48<img_buffer_width && PosY+48<img_height )
	{
		_img[_imgY+48][_imgX+48] = img[img_pos+__umul24(img_buffer_width, 48)+48];
	}
	__syncthreads();

	int sqsum_pos = __umul24(PosY, sqsum_buffer_width) + PosX;
	
	_sqsum[(_imgY<<5) + _imgX] = sqsum[sqsum_pos];
	_sqsum[(_imgY<<5) + _imgX+16] = sqsum[sqsum_pos + 16];
	_sqsum[((_imgY+16)<<5) + _imgX] = sqsum[sqsum_pos + (sqsum_buffer_width<<4)];
	_sqsum[((_imgY+16)<<5) + _imgX+16] = sqsum[sqsum_pos + (sqsum_buffer_width<<4) + 16];

	int lpos = (threadIdx.y<<4) + threadIdx.x;
	int _base_offset = __umul24(blockIdx.y, gridDim.x)+blockIdx.x;
	int basic_plan_pos = _base_offset<<9;
	unsigned int plan_pos = (lpos & 31) + ( ( (lpos & 192) >> 6 ) << 5);
	unsigned int arrayPos = ( lpos & 32 ) >> 5;
	
	for (int basic_plan_offset = 0; basic_plan_offset<512; basic_plan_offset+=128)
	{
		_res[lpos] = 0;
		if (lpos<128)
		{
			unsigned int *_planI = (unsigned int*)_plan;
			_planI[lpos] = plan[basic_plan_pos + basic_plan_offset + lpos];
		}
		__syncthreads();
		
		for (int res_base_pos = 0; res_base_pos<256; res_base_pos +=128 )
		{
			unsigned short plan_item = _plan[res_base_pos + plan_pos];
			unsigned char* cur_img = (unsigned char*)&_img[(plan_item>>5)][plan_item&0x1F];  
			float variance_norm_factor = _sqsum[plan_item];
			gpu_cascade* curr_cascade = haar_cascade;
			gpu_node* curr_node = &haar_nodes[arrayPos];
			for(int k=0;k<cascade_count;k++)
			{
				float stage_sum = 0.0f;
				if (plan_item <= 1024)
				{
					gpu_rect *curr_rec = &haar_rects2[curr_cascade[k].node2_first] + (arrayPos<<1);
					int j = curr_cascade[k].node2_first + (arrayPos<<1);
					for (int i = arrayPos; i<curr_cascade[k].node2_count; i+=2 )
					{
						float t = curr_node->threashold * variance_norm_factor;
						int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
						float r = sum*haar_rect_weights2[j];
						curr_rec++;
						j++;
						sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
						r += (sum*haar_rect_weights2[j]);
						stage_sum += r < t ? curr_node->a : curr_node->b;
						curr_rec += 3;
						j += 3;
						curr_node+=2;
					}
					//NODE2_CYCLE(2, 3, 3, 2);

					curr_node = &haar_nodes[arrayPos+curr_cascade[k].node2_count];
					curr_rec = &haar_rects3[curr_cascade[k].node3_first] + __umul24(arrayPos, 3);
					j = curr_cascade[k].node3_first + __umul24(arrayPos, 3);

					for (int i=arrayPos; i<curr_cascade[k].node3_count;i+=2 )
					{
						float t = curr_node->threashold * variance_norm_factor;
						int sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
						float r = sum*haar_rect_weights3[j];
						curr_rec++;
						j++;
						sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
						r += (sum*haar_rect_weights3[j]);
						curr_rec++;
						j++;
						sum =  *(unsigned int*)(cur_img + curr_rec->p1) - *(unsigned int*)(cur_img + curr_rec->p2) - *(unsigned int*)(cur_img + curr_rec->p3) + *(unsigned int*)(cur_img + curr_rec->p4);
						r += (sum*haar_rect_weights3[j]);
						stage_sum += r < t ? curr_node->a : curr_node->b;
						curr_rec+=4;
						j+=4;
						curr_node+=2;
					}
					//NODE3_CYCLE(2, 4, 4, 2);
					if ( (arrayPos) )
						_stage_sum[plan_pos] = stage_sum;
				}
				__syncthreads();
				if (arrayPos == 0 && plan_item <= 1024 )
				{
	   				stage_sum = _stage_sum[plan_pos] + stage_sum;
					if( stage_sum < curr_cascade[k].threashold )	
						_res[res_base_pos + plan_pos] = 0;
					else 
						_res[res_base_pos + plan_pos] = 255; 
				}
			}
		}			
		__syncthreads();
		unsigned int *_resI = (unsigned int *)_res;
		if (threadIdx.y<4)
		{
			int resPos =  (_base_offset<<8) + (threadIdx.y<<4) + threadIdx.x + (basic_plan_offset>>1);
			res[resPos] = _resI[lpos];
		}
		__syncthreads();
	}
}

#define PLAN_ITEM_BRANCH_4threads(base_offset) {											\
	PLAN_ITEM(4, 7, 10, writePos);															\
	if (plan_item <= 1024 && ( arrayPos & 1) == 0)											\
	{																						\
		stage_sum += _stage_sum [writePos];													\
		if (arrayPos == 2)																	\
			_stage_sum [plan_pos] = stage_sum;												\
	}																						\
	__syncthreads();																		\
	if (plan_item <= 1024 && arrayPos == 0)													\
	{																						\
		stage_sum = _stage_sum[plan_pos] + stage_sum;										\
		if( stage_sum < curr_cascade[0].threashold )										\
			_res[base_offset + plan_pos] = 0;												\
		else 																				\
			_res[base_offset + plan_pos] = 255; 											\
	}																						\
}																							\

__global__ static void haar_next_stage_4threads(unsigned int *img ,float* sqsum, unsigned int *res, 
									   unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
									   unsigned int sqsum_buffer_width, unsigned int cascade_count, unsigned int *plan)
{
	__shared__ unsigned int _img[52][52]; //(20+16) * (20+16)
	__shared__ float _sqsum[32*32];
	__shared__ unsigned char _res[256];
	__shared__ unsigned short _plan[256];
	__shared__ float _stage_sum[128];
	int _imgX = threadIdx.x;
	int _imgY = threadIdx.y;
	
	unsigned int PosX = (blockIdx.x<<5) + threadIdx.x;
	unsigned int PosY = (blockIdx.y<<5) + threadIdx.y;
	
	int img_pos = __umul24(PosY, img_buffer_width) + PosX;
	
	_img[_imgY][_imgX] = img[img_pos];
	_img[_imgY][_imgX+16] = img[img_pos+16];
	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];
	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];
	
	_img[_imgY][_imgX+32] = img[img_pos+32];
	_img[_imgY+16][_imgX+32] = img[img_pos+32+(img_buffer_width<<4)];
	_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];
	_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5) + 16];
	_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5) + 32];
	
	if (threadIdx.x<4 && PosX+48<img_buffer_width )
	{
		_img[_imgY][_imgX+48] = img[img_pos + 48];
		_img[_imgY+16][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<4)];
		_img[_imgY+32][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<5)];
	}
	if (threadIdx.y<4 && PosY+48<img_height)
	{
		_img[_imgY+48][_imgX] = img[img_pos+__umul24(img_buffer_width, 48)];
		_img[_imgY+48][_imgX+16] = img[img_pos+__umul24(img_buffer_width, 48)+16];
		_img[_imgY+48][_imgX+32] = img[img_pos+__umul24(img_buffer_width, 48)+32];
	}
	if (threadIdx.x<4 && threadIdx.y<4 && PosX+48<img_buffer_width && PosY+48<img_height )
	{
		_img[_imgY+48][_imgX+48] = img[img_pos+__umul24(img_buffer_width, 48)+48];
	}
	__syncthreads();

	int sqsum_pos = __umul24(PosY, sqsum_buffer_width) + PosX;
	
	_sqsum[(_imgY<<5) + _imgX] = sqsum[sqsum_pos];
	_sqsum[(_imgY<<5) + _imgX+16] = sqsum[sqsum_pos + 16];
	_sqsum[((_imgY+16)<<5) + _imgX] = sqsum[sqsum_pos + (sqsum_buffer_width<<4)];
	_sqsum[((_imgY+16)<<5) + _imgX+16] = sqsum[sqsum_pos + (sqsum_buffer_width<<4) + 16];
	
	int lpos = (threadIdx.y<<4) + threadIdx.x;
	int _base_offset = __umul24(blockIdx.y, gridDim.x)+blockIdx.x;
	int basic_plan_pos = _base_offset<<9;
	unsigned int plan_pos = (lpos & 31) + (((lpos & 128) >> 7)<<5);
	unsigned int arrayPos = (lpos & 96) >> 5;
	int writePos = ( ( (arrayPos & 2) >> 1 ) <<6 ) + plan_pos;
	
	for (int basic_plan_offset = 0; basic_plan_offset<512; basic_plan_offset+=128)
	{
		_res[lpos] = 0;
		if (lpos<128)
		{
			unsigned int *_planI = (unsigned int*)_plan;
			_planI[lpos] = plan[basic_plan_pos + basic_plan_offset + lpos];
		}
		__syncthreads();

		unsigned short plan_item = _plan[plan_pos];
		gpu_cascade* curr_cascade = haar_cascade;
		float stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_4threads(0);

		plan_item = _plan[plan_pos + 64];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_4threads(64);

		plan_item = _plan[plan_pos + 128];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_4threads(128);

		plan_item = _plan[plan_pos + 192];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_4threads(192);
		__syncthreads();
		
		unsigned int *_resI = (unsigned int *)_res;
		if (threadIdx.y<4)
		{
			int resPos =  (_base_offset<<8) + (threadIdx.y<<4) + threadIdx.x + (basic_plan_offset>>1);
			res[resPos] = _resI[lpos];
		}
		__syncthreads();
	}
}

#define PLAN_ITEM_BRANCH_8threads(base_offset) {											\
	PLAN_ITEM(8, 15, 22, writePos);															\
	if ( (arrayPos & 1) == 0 && plan_item <= 1024)											\
	{																						\
		int pos = (arrayPos & 7) >> 1;														\
		stage_sum += _stage_sum [writePos];													\
																							\
		if ( pos > 0)																		\
		_stage_sum [ ( (pos - 1) << 5 ) + plan_pos] = stage_sum;							\
	}																						\
	__syncthreads();																		\
																							\
	if ( arrayPos == 0 && plan_item <= 1024)												\
	{																						\
		stage_sum = _stage_sum[plan_pos] + _stage_sum[32 + plan_pos] + _stage_sum[64 + plan_pos] + stage_sum;	\
		if( stage_sum < curr_cascade[0].threashold )										\
			_res[base_offset + plan_pos] = 0;												\
		else 																				\
			_res[base_offset + plan_pos] = 255; 											\
	}																						\
}																							\

__global__ static void haar_next_stage_8threads(unsigned int *img ,float* sqsum, unsigned int *res, 
									   unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
									   unsigned int sqsum_buffer_width, unsigned int cascade_count, unsigned int *plan)
{
	__shared__ unsigned int _img[52][52]; //(20+16) * (20+16)
	__shared__ float _sqsum[32*32];
	__shared__ unsigned char _res[256];
	__shared__ unsigned short _plan[256];
	__shared__ float _stage_sum[128];
	int _imgX = threadIdx.x;
	int _imgY = threadIdx.y;
	
	unsigned int PosX = (blockIdx.x<<5) + threadIdx.x;
	unsigned int PosY = (blockIdx.y<<5) + threadIdx.y;
	
	int img_pos = __umul24(PosY, img_buffer_width) + PosX;
	
	_img[_imgY][_imgX] = img[img_pos];
	_img[_imgY][_imgX+16] = img[img_pos+16];
	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];
	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];
	
	_img[_imgY][_imgX+32] = img[img_pos+32];
	_img[_imgY+16][_imgX+32] = img[img_pos+32+(img_buffer_width<<4)];
	_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];
	_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5) + 16];
	_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5) + 32];
	
	if (threadIdx.x<4 && PosX+48<img_buffer_width )
	{
		_img[_imgY][_imgX+48] = img[img_pos + 48];
		_img[_imgY+16][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<4)];
		_img[_imgY+32][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<5)];
	}
	if (threadIdx.y<4 && PosY+48<img_height)
	{
		_img[_imgY+48][_imgX] = img[img_pos+__umul24(img_buffer_width, 48)];			// ИСПРАВЛЕНО. все 3 строки исходные (32 * ...) заменены на (48 * ...)
		_img[_imgY+48][_imgX+16] = img[img_pos+__umul24(img_buffer_width, 48)+16];
		_img[_imgY+48][_imgX+32] = img[img_pos+__umul24(img_buffer_width, 48)+32];
	}
	if (threadIdx.x<4 && threadIdx.y<4 && PosX+48<img_buffer_width && PosY+48<img_height )
	{
		_img[_imgY+48][_imgX+48] = img[img_pos+__umul24(img_buffer_width, 48)+48];
	}
	__syncthreads();

	int sqsum_pos = PosY*sqsum_buffer_width + PosX;
	
	_sqsum[(_imgY<<5) + _imgX] = sqsum[sqsum_pos];
	_sqsum[(_imgY<<5) + _imgX+16] = sqsum[sqsum_pos + 16];
	_sqsum[((_imgY+16)<<5) + _imgX] = sqsum[sqsum_pos + (sqsum_buffer_width<<4)];
	_sqsum[((_imgY+16)<<5) + _imgX+16] = sqsum[sqsum_pos + (sqsum_buffer_width<<4) + 16];

	int lpos = (threadIdx.y<<4) + threadIdx.x;
	int _base_offset = __umul24(blockIdx.y, gridDim.x)+blockIdx.x;
	int basic_plan_pos = _base_offset<<9;
	unsigned int plan_pos = lpos & 31;
	unsigned int arrayPos = (lpos & 224) >> 5;
	int writePos = ( ( (arrayPos & 6) >> 1 )<<5 ) + plan_pos;
	// раскрытие данного цикла значительно увеличивает время
	for (int basic_plan_offset = 0; basic_plan_offset<512; basic_plan_offset+=128)
	{
		_res[lpos] = 0;
		if (lpos<128)
		{
			unsigned int *_planI = (unsigned int*)_plan;
			_planI[lpos] = plan[basic_plan_pos + basic_plan_offset + lpos];
		}
		__syncthreads();
		
		unsigned short plan_item = _plan[plan_pos];
		gpu_cascade* curr_cascade = haar_cascade;
		float stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(0);
		
		plan_item = _plan[plan_pos + 32];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(32);
		
		plan_item = _plan[plan_pos + 64];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(64);
		
		plan_item = _plan[plan_pos + 96];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(96);
		
		plan_item = _plan[plan_pos + 128];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(128);
		
		plan_item = _plan[plan_pos + 160];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(160);
		
		plan_item = _plan[plan_pos + 192];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(192);
		
		plan_item = _plan[plan_pos + 224];
		stage_sum = 0.0f;
		PLAN_ITEM_BRANCH_8threads(224);
		__syncthreads();
		
		unsigned int *_resI = (unsigned int *)_res;
		if (threadIdx.y<4)
		{
			int resPos =  (_base_offset<<8) + (threadIdx.y<<4) + threadIdx.x + (basic_plan_offset>>1);
			res[resPos] = _resI[lpos];
		}
		__syncthreads();
	}
}

#define PROC_BLOCK_SIZE 1024
#define PROC_BLOCK_SIZE_LOG 10
#define PROC_BLOCK_SIZE4 256
#define PROC_BLOCK_SIZE4_LOG 8
#define THREADS 256
#define LOG_NUM_BANKS_SHORT 5

#define CONFLICT_FREE_OFFSET(index) (((index) >> LOG_NUM_BANKS_SHORT)<<1)

__device__  unsigned short scan(unsigned short *sum, unsigned short *len )
{
	int offset = 1;
	int thid  = threadIdx.x;
	// build the sum in place up the tree
	for (int d = THREADS; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)	  
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			sum[bi] += sum[ai];
		}

		offset = offset<<1 ;
	}

	// scan back down the tree

	// clear the last element
	if (thid == 0)
	{
		int index = 2*THREADS - 1;
		index += CONFLICT_FREE_OFFSET(index);
		*len = sum[index];
		sum[index] = 0;
	}   

	// traverse down the tree building the scan in place
	for (int d = 1; d < 2*THREADS; d <<= 1)
	{
		offset = offset>>1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned short t  = sum[ai];
			sum[ai] = sum[bi];
			sum[bi] += t;
		}
	}

	__syncthreads();
	return 0;
}

__global__ static void DataToQueueB1(unsigned int * device_src /*flags*/, unsigned int *device_result)
{
	__shared__ unsigned char src[PROC_BLOCK_SIZE];
	__shared__ unsigned short res[PROC_BLOCK_SIZE];
	__shared__ unsigned short sum[THREADS*2+32];
	__shared__ unsigned short len; 
	unsigned int *ps = (unsigned int *)src;
	unsigned int *pd = (unsigned int *)res;
	unsigned int *_device_result =  device_result;

	unsigned int PosX = (blockIdx.x<<1) +((threadIdx.x>>6)&0x1);
	unsigned int PosY = (blockIdx.y<<1) +(threadIdx.x>>7);
	unsigned int device_src_pos = ((__umul24(PosY, (gridDim.x<<1))+PosX)<<6) + ((threadIdx.x)&0x3F);
	
	/* threadIdx.x[7]*512 + threadIdx.x[5:2]*8 + threadIdx.x[6]*4 + threadIdx.x[1:0] */	
	unsigned int ps_pos = ((threadIdx.x & 0x80)) + ((threadIdx.x&0x3C)<<1) + ((threadIdx.x>>4)&0x4)  + (threadIdx.x & 0x3);  

	unsigned int tmp = device_src[device_src_pos]; 
	ps[ps_pos]  =  ((tmp&0x80808080)>>7);
	__syncthreads();
	
	int i2 = threadIdx.x<<1;
	int i4 = threadIdx.x<<2;
			
	int bankOffset = CONFLICT_FREE_OFFSET(i2);
	sum[i2 + bankOffset] = src[i4] + src[i4 + 1];
	sum[i2 + bankOffset + 1] = src[i4 + 2] + src[i4 + 3];
	res[i4] = 2048;
	res[i4+1] = 2048;
	res[i4+2] = 2048;
	res[i4+3] = 2048;
	__syncthreads();
	
	scan(sum, &len);
	
	unsigned int bp = sum[i2 + bankOffset];
	unsigned int c  = src[i4];
	if (src[i4])
		res[bp] = i4;
	
	if (src[i4 + 1])
		res[bp + c] = i4 + 1;
		
	bp = sum[i2 + bankOffset + 1];
	c  = src[i4 + 2];
	
	if (src[i4 + 2])
		res[bp] = i4 + 2;
		
	if (src[i4 + 3])
		res[bp + c] = i4 + 3;

	__syncthreads();
	unsigned int device_res_pos = ((__umul24(blockIdx.y, gridDim.x)+blockIdx.x)<<9) + threadIdx.x;
	_device_result[device_res_pos] = pd[threadIdx.x];
	_device_result[device_res_pos + 256] = pd[threadIdx.x + 256];
}

__global__ static void DataToQueueB(unsigned int * mask , unsigned int *old_plan, unsigned int *new_plan )
{
	__shared__ unsigned char src[PROC_BLOCK_SIZE];
	__shared__ unsigned short res[PROC_BLOCK_SIZE];
	__shared__ unsigned short _old_plan[PROC_BLOCK_SIZE];
	__shared__ unsigned short sum[THREADS*2+32];
	__shared__ unsigned short len; 
	unsigned int *ps = (unsigned int *)src;
	unsigned int *pd = (unsigned int *)res;
	
	int base_offset = __umul24(gridDim.x, blockIdx.y) + blockIdx.x;
	unsigned int mask_pos = (base_offset<<8) + (threadIdx.y<<4) + threadIdx.x;
	unsigned int tmp = mask[mask_pos]; 
	ps[threadIdx.x]  =  ((tmp&0x80808080)>>7);
	
	unsigned int old_plan_pos = (base_offset<<9) + (threadIdx.y<<4) + threadIdx.x;
	unsigned int *_old_planI = (unsigned int *)_old_plan;
	_old_planI[threadIdx.x] = old_plan[old_plan_pos];
	_old_planI[threadIdx.x+256] = old_plan[old_plan_pos+256];
	__syncthreads();

	int i2 = threadIdx.x<<1;
	int i4 = threadIdx.x<<2;
	int bankOffset = CONFLICT_FREE_OFFSET(i2);
	sum[i2 + bankOffset] = src[i4] + src[i4 + 1];
	sum[i2 + bankOffset + 1] = src[i4 + 2] + src[i4 + 3];
	res[i4] = 2048;
	res[i4+1] = 2048;
	res[i4+2] = 2048;
	res[i4+3] = 2048;
	__syncthreads();
	
	scan(sum, &len);
	unsigned int bp = sum[i2 + bankOffset];
	unsigned int c  = src[i4];
	if (src[i4])
		res[bp] = i4;
	if (src[i4 + 1])
		res[bp + c] = i4 + 1;
	bp = sum[i2 + bankOffset + 1];
	c  = src[i4 + 2];
	if (src[i4 + 2])
		res[bp] = i4 + 2;
	if (src[i4 + 3])
		res[bp + c] = i4 + 3;
	__syncthreads();

	unsigned int device_res_pos = (base_offset<<9) + threadIdx.x;
	tmp = 0;	
	unsigned int res_pos = threadIdx.x<<1;
	if (res[res_pos]<=1024)
		tmp = _old_plan[res[res_pos]];		 		
	else
		tmp = 2048;

	res_pos += 1;   
	if (res[res_pos]<=1024)
		tmp += (_old_plan[res[res_pos]]<<16);		 		
	else
		tmp += (2048<<16);
	new_plan[device_res_pos] = tmp;

	tmp = 0; 
	res_pos = (threadIdx.x<<1) + 512;   
	if (res[res_pos]<=1024)
		tmp = _old_plan[res[res_pos]];		 		
	else
		tmp = 2048;

	res_pos += 1;   
	if (res[res_pos]<=1024)
		tmp += (_old_plan[res[res_pos]]<<16);		 		
	else
		tmp += (2048<<16);

	new_plan[device_res_pos + 256] = tmp;
}

__global__ static void calcVarianceNormFactorKernel(unsigned int *img ,float* sqsum, float *res, 
											unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,
											unsigned int sqsum_buffer_width)
{
	__shared__ unsigned int _img[36][36]; //(20+16) * (20+16)
	__shared__ float _sqsum[16][16];

	int _imgX = threadIdx.x;
	int _imgY = threadIdx.y;

	unsigned int PosX = blockIdx.x*16 + threadIdx.x;
	unsigned int PosY = blockIdx.y*16 + threadIdx.y;

	int img_pos = PosY*img_buffer_width + PosX;

	_img[_imgY][_imgX] = img[img_pos];
	_img[_imgY][_imgX+16] = img[img_pos+16];
	_img[_imgY+16][_imgX] = img[img_pos+16*img_buffer_width];
	_img[_imgY+16][_imgX+16] = img[img_pos+16*img_buffer_width + 16];
	if (threadIdx.x<4 && PosX+32<img_buffer_width )
	{
		_img[_imgY][_imgX+32] = img[img_pos + 32];
		_img[_imgY+16][_imgX+32] = img[img_pos+32 + 16*img_buffer_width];
	}
	if (threadIdx.y<4 && PosY+32<img_height)
	{
		_img[_imgY+32][_imgX] = img[img_pos+32*img_buffer_width];
		_img[_imgY+32][_imgX+16] = img[img_pos+32*img_buffer_width+16];
	}
	if (threadIdx.x<4 && threadIdx.y<4 && PosX+32<img_buffer_width && PosY+32<img_height )
	{
		_img[_imgY+32][_imgX+32] = img[img_pos+32*img_buffer_width+32];
	}

	int sqsum_pos = (blockIdx.y*16+threadIdx.y)*sqsum_buffer_width + blockIdx.x*16 + threadIdx.x;
	_sqsum[_imgY][_imgX] = sqsum[sqsum_pos];

	__syncthreads();

	if ((PosX + window_STEPX + 3<img_width) && (PosY + window_STEPY + 3<img_height))
	{
		unsigned char* _cur_img = (unsigned char*)&_img[threadIdx.y + 1][threadIdx.x + 1];
		int sum = 	  *(unsigned int*)(_cur_img)
					- *(unsigned int*)(_cur_img + 4*window_STEPX)
					- *(unsigned int*)(_cur_img + 4*window_STEPY*36)
					+ *(unsigned int*)(_cur_img + 4*window_STEPY*36 + 4*window_STEPX);
		float mean = sum*isq;
		
		float variance_norm_factor = _sqsum[_imgY][_imgX]*isq - mean*mean;

		if ( variance_norm_factor >= 0. )
			variance_norm_factor = sqrtf(variance_norm_factor);
		else
			variance_norm_factor = 1.0;
		res[sqsum_pos] = variance_norm_factor;
	}
}

// генерирурет исходный текст функции ядра для заданного классификатора.
// за основу взят SetCascadeToGPU, но удалена инициализация константной памяти в GPU
// Результат сохраняется в gkernel.cu
void generateCodeSequence(CvHaarClassifierCascade* cascade,int first_cascade, int cascade_count, int gpu_window_width)
{
	if (cascade->count < first_cascade || cascade->count < first_cascade + cascade_count)
		return;
	CvHaarStageClassifier *stages = &cascade->stage_classifier[first_cascade];

	int node2_count = 0;
	int node3_count = 0;
	for (int i=0; i<cascade_count;i++)
	{
		CvHaarClassifier* classifier = stages[i].classifier;
		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier->count != 1)
				return;
			if (classifier[j].haar_feature->rect[2].weight==0)
				node2_count++;
			else
				node3_count++;
		}
	}
	if (cascade_count>NODES || 2*node2_count>RECTS2 || 3*node3_count>RECTS3 || node2_count + node3_count > NODES)
		return;
	
	gpu_rect* _gpu_rect2s = (gpu_rect*)malloc(sizeof(gpu_rect)*2*node2_count);
	gpu_rect* _gpu_rect3s = (gpu_rect*)malloc(sizeof(gpu_rect)*3*node3_count);
	float* _haar_rect2_weights = (float*)malloc(sizeof(float)*node2_count*2);
	float* _haar_rect3_weights = (float*)malloc(sizeof(float)*node3_count*3);
	gpu_node* _haar_nodes = (gpu_node*)malloc(sizeof(gpu_node)*(node2_count + node3_count));
	gpu_cascade * _haar_cascade = (gpu_cascade *)malloc(sizeof(gpu_cascade)*cascade_count);
	
	gpu_window_width = cascade->orig_window_size.width + gpu_window_width;  //16+20 || 32+20
	const float icv_stage_threshold_bias = 0.0001f;

	int node2_pos = 0, node3_pos = 0, node_pos = 0;
	for (int i=0; i<cascade_count; i++)
	{
		int node2_count = 0, node3_count = 0;
		_haar_cascade[i].threashold = stages[i].threshold - icv_stage_threshold_bias;
		
		_haar_cascade[i].node2_first = node2_pos; //cascade->stage_classifier[i].two_rects == true ? node2_pos : -1;
		_haar_cascade[i].node3_first = node3_pos; //cascade->stage_classifier[i].two_rects == true ? node3_pos : -1; //node3_pos;
		 
		CvHaarClassifier* classifier = stages[i].classifier;
		
		for (int j=0;j<stages[i].count;j++)
		{
			   if (classifier[j].haar_feature->rect[2].weight == 0)
			   {
					CvRect *curr_rect = &classifier[j].haar_feature->rect[0].r;

					int x = curr_rect->x;
					int y = curr_rect->y;
					_gpu_rect2s[node2_pos].p1 = 4*(y * gpu_window_width + x); 
					_gpu_rect2s[node2_pos].p2 = 4*(y * gpu_window_width + x + curr_rect->width);
					_gpu_rect2s[node2_pos].p3 = 4*((y + curr_rect->height) * gpu_window_width + x);
					_gpu_rect2s[node2_pos].p4 =4*((y + curr_rect->height) * gpu_window_width + x + curr_rect->width);
					_haar_rect2_weights[node2_pos] = classifier[j].haar_feature->rect[0].weight * isq;
					
					curr_rect = &classifier[j].haar_feature->rect[1].r;
					node2_pos++;
					x = curr_rect->x;
					y = curr_rect->y;
					_gpu_rect2s[node2_pos].p1 = 4*(y * gpu_window_width + x); 
					_gpu_rect2s[node2_pos].p2 = 4*(y * gpu_window_width + x + curr_rect->width);
					_gpu_rect2s[node2_pos].p3 = 4*((y + curr_rect->height) * gpu_window_width + x);
					_gpu_rect2s[node2_pos].p4 =4*((y + curr_rect->height) * gpu_window_width + x + curr_rect->width);
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
					_gpu_rect3s[node3_pos].p1 = 4*(y * gpu_window_width + x); 
					_gpu_rect3s[node3_pos].p2 = 4*(y * gpu_window_width + x + curr_rect->width);
					_gpu_rect3s[node3_pos].p3 = 4*((y + curr_rect->height) * gpu_window_width + x);
					_gpu_rect3s[node3_pos].p4 =4*((y + curr_rect->height) * gpu_window_width + x + curr_rect->width);
					_haar_rect3_weights[node3_pos] = classifier[j].haar_feature->rect[0].weight * isq;
					 node3_pos++;
					curr_rect = &classifier[j].haar_feature->rect[1].r;
					x = curr_rect->x;
					y = curr_rect->y;
					_gpu_rect3s[node3_pos].p1 = 4*(y * gpu_window_width + x); 
					_gpu_rect3s[node3_pos].p2 = 4*(y * gpu_window_width + x + curr_rect->width);
					_gpu_rect3s[node3_pos].p3 = 4*((y + curr_rect->height) * gpu_window_width + x);
					_gpu_rect3s[node3_pos].p4 =4*((y + curr_rect->height) * gpu_window_width + x + curr_rect->width);
					_haar_rect3_weights[node3_pos] = classifier[j].haar_feature->rect[1].weight * isq;
					node3_pos++;  
					curr_rect = &classifier[j].haar_feature->rect[2].r;
					x = curr_rect->x;
					y = curr_rect->y;
					_gpu_rect3s[node3_pos].p1 = 4*(y * gpu_window_width + x); 
					_gpu_rect3s[node3_pos].p2 = 4*(y * gpu_window_width + x + curr_rect->width);
					_gpu_rect3s[node3_pos].p3 = 4*((y + curr_rect->height) * gpu_window_width + x);
					_gpu_rect3s[node3_pos].p4 =4*((y + curr_rect->height) * gpu_window_width + x + curr_rect->width);
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
	
	FILE *fp = fopen("gkernel.cu","w");
	if(fp == 0)
		return;
	
	fprintf(fp, "			unsigned char* cur_img = (unsigned char*)&_img[threadIdx.y][threadIdx.x];\n");
	fprintf(fp, "			int sum;\n");
	//fprintf(fp, "			gpu_cascade* curr_cascade = haar_cascade;\n");
	//fprintf(fp, "			gpu_node* curr_node = &haar_nodes[0];\n");
	fprintf(fp, "			for (int k = 0; k < 1; k++)\n");
	fprintf(fp, "			{\n");
	fprintf(fp, "				float stage_sum = 0.0f;\n");
	
	gpu_cascade* curr_cascade_ = _haar_cascade;
	gpu_node* curr_node_ = &_haar_nodes[0];
	for (int k = 0; k < cascade_count; k++)
	{
		gpu_rect *curr_rec_ = &_gpu_rect2s[curr_cascade_[k].node2_first];
		int j = curr_cascade_[k].node2_first;
		for (int i = 0; i < curr_cascade_[k].node2_count; i++ )
		{
			if (i == 0 && k == 0)
			{
				fprintf(fp, "				float r = 0.0f;\n");
				fprintf(fp, "				float t = %ff * _sqsum[_imgY][_imgX];\n", curr_node_->threashold);
			}
			else
			{
				fprintf(fp, "				r = 0.0f;\n");
				fprintf(fp, "				t = %ff * _sqsum[_imgY][_imgX];\n", curr_node_->threashold);
			}
			fprintf(fp, "				sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
						curr_rec_->p1, curr_rec_->p2, curr_rec_->p3, curr_rec_->p4);
			fprintf(fp, "				r = %ff*sum;\n",_haar_rect2_weights[j]);
			
			curr_rec_++;
			j++;
			
			fprintf(fp, "				sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
						curr_rec_->p1, curr_rec_->p2, curr_rec_->p3, curr_rec_->p4);
			fprintf(fp, "				r += %ff*sum;\n",_haar_rect2_weights[j]);
			fprintf(fp, "				stage_sum += r < t ? %ff: %f;\n\n", curr_node_->a, curr_node_->b);
			
			curr_rec_++;
			j++;
			curr_node_++;
		}
		
		curr_rec_ = &_gpu_rect3s[curr_cascade_[k].node3_first];
		j = curr_cascade_[k].node3_first;

		for (int i=0; i<curr_cascade_[k].node3_count;i++ )
		{
			fprintf(fp, "				r = 0.0f;\n");
			fprintf(fp, "				t = %ff * _sqsum[_imgY][_imgX];\n", curr_node_->threashold);
			fprintf(fp, "				sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
						curr_rec_->p1, curr_rec_->p2, curr_rec_->p3, curr_rec_->p4);
			fprintf(fp, "				r = %ff*sum;\n",_haar_rect3_weights[j]);
			
			curr_rec_++;
			j++;
			
			fprintf(fp, "				sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
						curr_rec_->p1, curr_rec_->p2, curr_rec_->p3, curr_rec_->p4);
			fprintf(fp, "				r += %ff*sum;\n",_haar_rect3_weights[j]);
			
			curr_rec_++;
			j++;
			
			fprintf(fp, "				sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
						curr_rec_->p1, curr_rec_->p2, curr_rec_->p3, curr_rec_->p4);
			fprintf(fp, "				r += %ff*sum;\n",_haar_rect3_weights[j]);
			fprintf(fp, "				stage_sum += r < t ? %ff: %f;\n\n", curr_node_->a, curr_node_->b);
			
			curr_rec_++;
			j++;
			curr_node_++;
		}
		fprintf(fp, "				if( stage_sum < %ff )\n", curr_cascade_[k].threashold);
		fprintf(fp, "				{\n");
		fprintf(fp, "					_res[res_pos] = 0;\n");
		fprintf(fp, "					break;\n");
		fprintf(fp, "				}\n");
		fprintf(fp, "				else\n");
		fprintf(fp, "					_res[res_pos] = 255;\n");
	}
	fprintf(fp, "			}\n");
	fclose(fp);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	free(_gpu_rect2s);
	free(_gpu_rect3s);
	free (_haar_rect2_weights) ;
	free (_haar_rect3_weights);
	free (_haar_nodes);
	free (_haar_cascade);
}

// инициализирует внутреннее представление каскада
int SetCascadeToGPU(CvHaarClassifierCascade* cascade,int first_cascade, int cascade_count, int gpu_window_width)
{
	if (cascade->count < first_cascade || cascade->count < first_cascade + cascade_count)
		return 0;
	CvHaarStageClassifier *stages = &cascade->stage_classifier[first_cascade];

	int node2_count = 0;
	int node3_count = 0;
	for (int i=0; i<cascade_count;i++)
	{
		CvHaarClassifier* classifier = stages[i].classifier;
		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier->count != 1)
				return 0;
			if (classifier[j].haar_feature->rect[2].weight==0)
				node2_count++;
			else
				node3_count++;
		}
	}
	if (cascade_count>NODES || (node2_count<<1)>RECTS2 || 3*node3_count>RECTS3 || node2_count + node3_count > NODES)
		return -1;

	gpu_rect* _gpu_rect2s = (gpu_rect*)malloc(sizeof(gpu_rect)*(node2_count<<1));
	gpu_rect* _gpu_rect3s = (gpu_rect*)malloc(sizeof(gpu_rect)*3*node3_count);
	float* _haar_rect2_weights = (float*)malloc(sizeof(float)*(node2_count<<1));
	float* _haar_rect3_weights = (float*)malloc(sizeof(float)*node3_count*3);

	gpu_node* _haar_nodes = (gpu_node*)malloc(sizeof(gpu_node)*(node2_count + node3_count));
	gpu_cascade * _haar_cascade = (gpu_cascade *)malloc(sizeof(gpu_cascade)*cascade_count);

	gpu_window_width = cascade->orig_window_size.width + gpu_window_width;  //16+20 || 32+20
	const float icv_stage_threshold_bias = 0.0001f;

	int node2_pos = 0, node3_pos = 0, node_pos = 0;
	for (int i=0; i<cascade_count; i++)
	{
		int node2_count = 0, node3_count = 0;
		_haar_cascade[i].threashold = stages[i].threshold - icv_stage_threshold_bias;
		_haar_cascade[i].node2_first = node2_pos;
		_haar_cascade[i].node3_first = node3_pos;
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

int iAlignUp(int a, int b)
{
	return (a % b != 0) ?  (a - a % b + b) : a;
}

// функция вызова ядра расчета variance_norm_factor
void calcVarianceNormFactor(unsigned int *integral_img, double *sqintegral_img, unsigned char* res_mask, int integral_img_width, int integral_img_height, char* res_fname)
{
	float *sqsum_img_gpu 			= NULL;	// указатель на память в GPU, хранящий sqsum
	unsigned int *integral_img_gpu	= NULL;	// указатель на память в GPU, хранящий интегральную сумму
	unsigned char *resultates		= NULL;	// указатель на память в GPU, хранящий результат работы классификатора
	float *variance_gpu				= NULL;	// указатель на память в GPU, хранящий variance_norm_factor

	int integral_img_width16 = iAlignUp(integral_img_width, 16);
	int integral_img_height16 = iAlignUp(integral_img_height, 16);
	
	// расчет интегральной суммы
	unsigned int *_integral_img = (unsigned int *)malloc (integral_img_width16*integral_img_height16*sizeof(unsigned int));
	memset(_integral_img,0, integral_img_width16*integral_img_height16*sizeof(unsigned int));
	for (int i=0;i<integral_img_height;i++)
		for (int j=0;j<integral_img_width;j++)
			_integral_img[integral_img_width16*i + j] = integral_img[integral_img_width*i + j];

	int wsizew = 20, wsizeh = 20;
	int windows_width =  integral_img_width - wsizew;
	int windows_height = integral_img_height-wsizeh;
	int windows_width16 = iAlignUp(windows_width, 16);
	int windows_height16 = iAlignUp(windows_height, 16);
	int window_stepX = wsizew - 2;
	int window_stepY = wsizeh - 2;
	
	unsigned int timer = 0;
	cutilCheckError( cutCreateTimer( &timer ) );

	// расчет sqsum
	float *sqsum_img = (float *)malloc (windows_width16*windows_height16*sizeof(float));
	memset(sqsum_img,0, windows_width16*windows_height16*sizeof(float));

	for (int i=1;i<windows_height;i++)
		for (int j=1;j<windows_width;j++)
		{
			sqsum_img[windows_width16*(i-1)+(j-1)] = sqintegral_img[i*integral_img_width + j] - sqintegral_img[(i)*integral_img_width + j + window_stepX]
					- sqintegral_img[(i+window_stepY)*integral_img_width + j] + sqintegral_img[(i+window_stepY)*integral_img_width + j + window_stepX];
		}

	float elapsedTimeInMs = 0.0f, elapsedTimeInMs2 = 0.0f;	

	cudaEvent_t start, stop;
	
	cutilSafeCall  ( cudaEventCreate( &start ) );
	cutilSafeCall  ( cudaEventCreate( &stop ) );

	int blockDimX = windows_width16/16;
	int blockDimY = iAlignUp(windows_height16,16)/16;
	cutilCheckError( cutStartTimer( timer));
	
	// выделяем и инициализируем все память
	cutilSafeCall(cudaMalloc((void**)&sqsum_img_gpu, 	sizeof(float) * windows_width16 * windows_height16));
	cutilSafeCall(cudaMalloc((void**)&variance_gpu, 	sizeof(float) * blockDimX * blockDimY * 256));
	cutilSafeCall(cudaMalloc((void**)&integral_img_gpu, sizeof(unsigned int) * integral_img_width16 * integral_img_height16));
	cutilSafeCall(cudaMalloc((void**)&resultates, 		sizeof(unsigned char) * blockDimX * blockDimY * 256));

	cutilSafeCall(cudaMemcpy(integral_img_gpu, _integral_img, sizeof(unsigned int) * integral_img_width16 * integral_img_height16, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(sqsum_img_gpu, sqsum_img, sizeof(float) *  windows_width16 * windows_height16, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemset(variance_gpu, 0,  sizeof(float) * blockDimX * blockDimY * 256));
	cutilSafeCall(cudaMemset(resultates, 0,  sizeof(unsigned char) * blockDimX * blockDimY * 256));

	dim3 threads = dim3(16, 16);
	dim3 blocks  = dim3(blockDimX, blockDimY);
	cutilSafeCall( cudaEventRecord( start, 0 ) );

	for (int i = 0; i< 1; i++)	
	{
		calcVarianceNormFactorKernel<<<blocks, threads>>> (integral_img_gpu, sqsum_img_gpu, variance_gpu, 
											integral_img_width-1, integral_img_height-1, integral_img_width16,
											windows_width16);
	}
	
	cutilSafeCall( cudaEventRecord( stop, 0 ) );
	cutilSafeCall( cudaThreadSynchronize() );   
	cutilCheckError( cutStopTimer( timer));

	cutilSafeCall( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
	elapsedTimeInMs2 = cutGetTimerValue( timer);
	printf("variance_norm_factor calulate:\n");
	printf ("Time 1 %f ms\n",elapsedTimeInMs );
	printf ("Time 2 %f ms\n",elapsedTimeInMs2 );
	
	cutilSafeCall( cudaEventDestroy(stop) );
	cutilSafeCall( cudaEventDestroy(start) );
	cutilCheckError( cutDeleteTimer( timer));
	free(sqsum_img);
	free(_integral_img);
}

void RunHaarClassifierCascade(CvHaarClassifierCascade* cascade,int first_cascade, int cascade_count,  int gpu_window_width, 
							   unsigned int *integral_img, double *sqintegral_img,unsigned char* res_mask, int integral_img_width,
							   int integral_img_height, char* res_fname)
{	
	const int wsizew = 20, wsizeh = 20;		
	int integral_img_width32 = iAlignUp(integral_img_width, 32);
	int integral_img_height32 = iAlignUp(integral_img_height, 32);
	
	int integral_img_buffer_width = integral_img_width32 + 32;
	int integral_img_buffer_height = integral_img_height32 + wsizeh;
	unsigned int *_integral_img = (unsigned int *)malloc (integral_img_buffer_width*integral_img_buffer_height*sizeof(unsigned int));

	memset(_integral_img,0, integral_img_buffer_width*integral_img_buffer_height*sizeof(unsigned int));
	for (int i=0;i<integral_img_height;i++)
		for (int j=0;j<integral_img_width;j++)
			_integral_img[integral_img_buffer_width*i + j] = integral_img[integral_img_width*i + j];

	int windows_width =  integral_img_width-wsizew;
	int windows_height = integral_img_height-wsizeh;
	int windows_width32 = iAlignUp(windows_width, 32);
	int windows_height32 = iAlignUp(windows_height, 32);
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

	float *varince_result = (float *)malloc (windows_width32*windows_height*sizeof(float));
	memset(varince_result,0, windows_width32*windows_height*sizeof(float));
	const double tmp = 1.0/(18.0*18.0);

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
	
	float *sqsum_img_gpu;
	unsigned int *integral_img_gpu;
	unsigned char *res_gpu, *res_cpu;
	unsigned short *plans_gpu, *_plans_gpu, *plans_cpu/*, *plan_old*/; //plan_old м.б. удален
	//DEBUG_INFO *result_gpu, *result_cpu;		// оба элемента могут быть удалены
	
	int blockDimX = (windows_width32>>4);
	int blockDimY = (windows_height32>>4);
	
	int blockDimX32 = (windows_width32>>5);
	int blockDimY32 = (windows_height32>>5);

#ifdef DEBUG_MODE
	printf("blockDimY %d blockDimX %d\n", blockDimY, blockDimX);
#endif

	int plans_length 	= sizeof(unsigned short)	* blockDimX32		* (blockDimY32 << 10);
	int res_length 		= sizeof(unsigned char) 	* blockDimX 		* (blockDimY << 8);
	int sqsum_length 	= sizeof(float) 			* windows_width32	* windows_height;
	int integralImg_length = integral_img_buffer_width * integral_img_buffer_height * sizeof(unsigned int);
	
	res_cpu = (unsigned char*)malloc( res_length );
	plans_cpu = (unsigned short*)malloc( plans_length );
	//plan_old = (unsigned short*)malloc(sizeof(unsigned short) * blockDimX32 * blockDimY32 * 1024);
	//result_cpu = (DEBUG_INFO*)malloc(sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32);
	//memset(result_cpu,0,sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32);
	
	cutilSafeCall(cudaMalloc((void**)&sqsum_img_gpu,	sqsum_length));
	cutilSafeCall(cudaMalloc((void**)&res_gpu,			res_length ));
	cutilSafeCall(cudaMalloc((void**)&integral_img_gpu,	integralImg_length));
	cutilSafeCall(cudaMalloc((void**)&plans_gpu,		plans_length));
	cutilSafeCall(cudaMalloc((void**)&_plans_gpu,		plans_length));
	//cutilSafeCall(cudaMalloc((void**)&result_gpu, sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32));
	
	cutilSafeCall(cudaMemcpy(integral_img_gpu, _integral_img,  integralImg_length,	cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(sqsum_img_gpu,    varince_result, sizeof(float) * windows_width32 * windows_height, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemset(res_gpu,    0, res_length));
	cutilSafeCall(cudaMemset(plans_gpu,  0, plans_length));
	cutilSafeCall(cudaMemset(_plans_gpu, 0, plans_length));
	//cutilSafeCall(cudaMemset(result_gpu, 0, sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32));
	
	if (SetCascadeToGPU(cascade, 0, 2, 16) < 0)
		printf("Couldn't initialize cascade\n");

	dim3 threads 	= dim3(16, 16);
	dim3 threads256 = dim3(256);
	dim3 blocks  	= dim3(blockDimX, blockDimY);
	dim3 blocks32 	= dim3(blockDimX32, blockDimY32);

#ifdef DEBUG_MODE
	cutilCheckError( cutStartTimer  ( timer ) );	 
	cutilSafeCall  ( cudaEventRecord( start, 0 ) );	
#endif

	haar_first_stage<<<blocks, threads>>> ( integral_img_gpu, sqsum_img_gpu, (unsigned int*)res_gpu, 
											integral_img_width-1, integral_img_height-1, integral_img_buffer_width,
											windows_width32, 2);
	DataToQueueB1<<<blocks32,threads256>>>  ((unsigned int*)res_gpu, (unsigned int*)plans_gpu);

	if (SetCascadeToGPU(cascade, 2, 1, 32) < 0)
		printf("Couldn't initialize cascade\n");

	haar_next_stage<<<blocks32, threads>>> (integral_img_gpu, sqsum_img_gpu, (unsigned int*)res_gpu, 
											integral_img_width-1, integral_img_height-1, integral_img_buffer_width,
											windows_width32, 1, (unsigned int*)plans_gpu/*, result_gpu, 0*/);	
	/*{	cutilSafeCall( cudaMemcpy(result_cpu, result_gpu,  sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32, cudaMemcpyDeviceToHost)); 
		cutilSafeCall(cudaMemset(result_gpu, 0, sizeof(DEBUG_INFO) * 256 * blockDimX32 * blockDimY32));
		char str[256];
		sprintf(str,"Debug_%d_EMU.txt", 0);
		FILE *fp = fopen(str,"wb");
		if (fp!=0)
		{
			//for (int blocksX = 0; blocksX < blocks.x; blocksX++)
			//{
			//	fprintf(fp,"blockIdx.x: %d\n", blocksX);
				for (int threads = 0; threads < 128; threads++)
				{
					fprintf(fp,"\t\tthreadIdx: %d %d\n", threads / 16, threads % 16);
					
					DEBUG_INFO *debug = &result_cpu[threads];
					for (int cascades = 0; cascades < debug->cascade_count; cascades++)
					{
						fprintf(fp, "cascade = %d\n", cascades);
						fprintf(fp, "\tvariance_norm_factor = %.10f\n", debug->test[cascades].variance_norm_factor);
						fprintf(fp, "\tNode 2 elements:\n");
						for (int node2_index = 0; node2_index < debug->test[cascades].node2_count; node2_index++)
						{
							fprintf(fp, "\t\tthreashold = %.10f\n", debug->test[cascades].node2[node2_index].threashold);
							fprintf(fp, "\t\tt = %.10f\n", debug->test[cascades].node2[node2_index].t);
							
							fprintf(fp, "\t\tsum1 = %d\n", debug->test[cascades].node2[node2_index].sum1);
							fprintf(fp, "\t\tweights1 = %.10f\n", debug->test[cascades].node2[node2_index].rect_weights1);
							fprintf(fp, "\t\tr1 = %.10f\n", debug->test[cascades].node2[node2_index].r1);
							
							fprintf(fp, "\t\tsum2 = %d\n", debug->test[cascades].node2[node2_index].sum2);
							fprintf(fp, "\t\tweights2 = %.10f\n", debug->test[cascades].node2[node2_index].rect_weights2);
							fprintf(fp, "\t\tr2 = %.10f\n", debug->test[cascades].node2[node2_index].r2);
							
							fprintf(fp, "\t\tstage_sum = %.10f\n", debug->test[cascades].node2[node2_index].stage_sum);
							fprintf(fp, "\t\ta = %.10f\n", debug->test[cascades].node2[node2_index].a);
							fprintf(fp, "\t\tb = %.10f\n", debug->test[cascades].node2[node2_index].b);
						}
						//fprintf(fp, "\t\t\tstage_sum after node2 (1) = %.10f\n", debug->test[cascades].stage_sum_after_node2[0]);
						//fprintf(fp, "\t\t\tstage_sum after node2 (2) = %.10f\n", debug->test[cascades].stage_sum_after_node2[1]);
						
						fprintf(fp, "\tNode 3 elements:\n");
						for (int node3_index = 0; node3_index < debug->test[cascades].node3_count; node3_index++)
						{
							fprintf(fp, "\t\tthreashold = %.10f\n", debug->test[cascades].node3[node3_index].threashold);
							fprintf(fp, "\t\tt = %.10f\n", debug->test[cascades].node3[node3_index].t);
							
							fprintf(fp, "\t\tsum1 = %d\n", debug->test[cascades].node3[node3_index].sum1);
							fprintf(fp, "\t\tweights1 = %.10f\n", debug->test[cascades].node3[node3_index].rect_weights1);
							fprintf(fp, "\t\tr1 = %.10f\n", debug->test[cascades].node3[node3_index].r1);
							
							fprintf(fp, "\t\tsum2 = %d\n", debug->test[cascades].node3[node3_index].sum2);
							fprintf(fp, "\t\tweights2 = %.10f\n", debug->test[cascades].node3[node3_index].rect_weights2);
							fprintf(fp, "\t\tr2 = %.10f\n", debug->test[cascades].node3[node3_index].r2);
							
							fprintf(fp, "\t\tsum3 = %d\n", debug->test[cascades].node3[node3_index].sum3);
							fprintf(fp, "\t\tweights3 = %.10f\n", debug->test[cascades].node3[node3_index].rect_weights3);
							fprintf(fp, "\t\tr3 = %.10f\n", debug->test[cascades].node3[node3_index].r3);
							
							fprintf(fp, "\t\tstage_sum = %.10f\n", debug->test[cascades].node3[node3_index].stage_sum);
							fprintf(fp, "\t\ta = %.10f\n", debug->test[cascades].node3[node3_index].a);
							fprintf(fp, "\t\tb = %.10f\n", debug->test[cascades].node3[node3_index].b);
						}
						
						fprintf(fp, "\t\t\tstage_sum = %.10f\n", debug->test[cascades].stage_sum);
						fprintf(fp, "\t\t\tthreashold = %.10f\n\n", debug->test[cascades].threashold);
					}
				}
				fprintf(fp,"-------------------------\n");
			//}
			fclose(fp);
		}
	}*/
	// берем каскады, которые лучше всего рассчитываются на 2-поточной версии
	unsigned short *plans_gpu_tmp;
	for (int i = 3; i < 8; i++)
	{
		DataToQueueB<<<blocks32,threads256>>>  ((unsigned int*)res_gpu, (unsigned int*)plans_gpu, (unsigned int*)_plans_gpu);
		if (SetCascadeToGPU(cascade, i, 1, 32) < 0)
		{
			printf("Couldn't initialize cascade\n");
			break;
		}
		haar_next_stage<<<blocks32, threads>>> (integral_img_gpu, sqsum_img_gpu, (unsigned int*)res_gpu, 
							integral_img_width-1, integral_img_height-1, integral_img_buffer_width,
							windows_width32, 1, (unsigned int*)_plans_gpu);
		// копируем во временный массив данные для последующей обработки в DataToQueueB
		plans_gpu_tmp = plans_gpu;
		plans_gpu = _plans_gpu;
		_plans_gpu = plans_gpu_tmp;
		//cutilSafeCall( cudaMemcpy(plans_gpu, _plans_gpu,  plans_length, cudaMemcpyDeviceToDevice));
	}
	// берем каскады, которые лучше всего рассчитываются на 4-поточной версии,
	//		кроме последнего (13 каскад), т.к. для него не обязательно выполнять копирование в plans_gpu.
	for (int i = 8; i < cascade_count - 1; i++)
	{
		DataToQueueB<<<blocks32,threads256>>>  ((unsigned int*)res_gpu, (unsigned int*)plans_gpu, (unsigned int*)_plans_gpu);
		if (SetCascadeToGPU(cascade, i, 1, 32) < 0)
		{
			printf("Couldn't initialize cascade\n");
			break;
		}
		haar_next_stage_4threads<<<blocks32, threads>>> (integral_img_gpu, sqsum_img_gpu, (unsigned int*)res_gpu, 
								integral_img_width-1, integral_img_height-1, integral_img_buffer_width,
								windows_width32, 1, (unsigned int*)_plans_gpu);
		// копируем во временный массив данные для последующей обработки в DataToQueueB
		//cutilSafeCall( cudaMemcpy(plans_gpu, _plans_gpu,  plans_length, cudaMemcpyDeviceToDevice));
		plans_gpu_tmp = plans_gpu;
		plans_gpu = _plans_gpu;
		_plans_gpu = plans_gpu_tmp;
	}

	DataToQueueB<<<blocks32,threads256>>>  ((unsigned int*)res_gpu, (unsigned int*)plans_gpu, (unsigned int*)_plans_gpu);
	if (SetCascadeToGPU(cascade, cascade_count - 1, 1, 32) < 0)
		printf("Couldn't initialize cascade\n");
	haar_next_stage_4threads<<<blocks32, threads>>> (integral_img_gpu, sqsum_img_gpu, (unsigned int*)res_gpu, 
							integral_img_width-1, integral_img_height-1, integral_img_buffer_width,
							windows_width32, 1, (unsigned int*)_plans_gpu);

#ifdef DEBUG_MODE
	cutilSafeCall( cudaEventRecord( stop, 0 ) );
	cutilSafeCall( cudaThreadSynchronize() );
	cutilCheckError( cutStopTimer( timer));
#endif

	cutilSafeCall( cudaMemcpy(res_cpu, 	 res_gpu,  	 res_length,   cudaMemcpyDeviceToHost)); 
	cutilSafeCall( cudaMemcpy(plans_cpu, _plans_gpu, plans_length, cudaMemcpyDeviceToHost)); 
	//cutilSafeCall( cudaMemcpy(plan_old, plans_gpu,  sizeof(unsigned short) * blockDimX32 * blockDimY32 * 1024,cudaMemcpyDeviceToHost)); 

#ifdef DEBUG_MODE
	cutilSafeCall( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
	elapsedTimeInMs2 = cutGetTimerValue( timer);
	printf ("Time 1 %f ms\n",elapsedTimeInMs );
	printf ("Time 2 %f ms\n",elapsedTimeInMs2 );
#endif
	
	windows_width -= 1;
	windows_height -= 1;
  /*  
	for (int i=0; i<blockDimY;i++ )
		for (int j=0; j<blockDimX;j++ )
		{
				for (int k=0;k<16;k++)
					for (int l=0;l<16;l++)	
					{
						int offset  = (k*16 + l);
						//int offset  = (k*16 + l)*4;
						//offset = (offset>>8) + offset&0xFF;
						if (i*16+k < windows_height && j*16 + l < windows_width )
							res_mask[i*windows_width*16+j*16+k*windows_width+l] = res_cpu[256*(i*blockDimX+j) + offset];
					}
		}
 
 
	for (int i=0; i<blockDimY32;i++ )
		for (int j=0; j<blockDimX32;j++ )
		{
				for (int k=0;k<32;k++)
					for (int l=0;l<32;l++)	
					{
						int offset  = (k*32 + l);	// 0..1023
						//int offset  = (k*16 + l)*4;
						//offset = (offset>>8) + offset&0xFF;
						if (i*32+k < windows_height && j*32 + l < windows_width )
							res_mask[i*windows_width*32+j*32+k*windows_width+l] = res_cpu[1024*(i*blockDimX32+j) + offset];
					}
		}
*/

	cutilSafeCall(cudaFree(sqsum_img_gpu));
	cutilSafeCall(cudaFree(integral_img_gpu));
	cutilSafeCall(cudaFree(res_gpu));
	cutilSafeCall(cudaFree(plans_gpu));
	cutilSafeCall(cudaFree(_plans_gpu));
	//cutilSafeCall(cudaFree(result_gpu));

#ifdef DEBUG_MODE
	cutilSafeCall( cudaEventDestroy(stop) );
	cutilSafeCall( cudaEventDestroy(start) );
	cutilCheckError( cutDeleteTimer( timer));
#endif

	//unsigned int *dump_plans_cpu = (unsigned int *)malloc(sizeof(unsigned int) * blockDimX32 * blockDimY32 * 1024);
	//unsigned int *dump_full_mask_cpu = (unsigned int *)malloc(sizeof(unsigned int) * windows_width * windows_height);
	//memset (dump_full_mask_cpu, 0 , sizeof(unsigned int) * windows_width * windows_height);
	//memset (dump_plans_cpu, 0 , sizeof(unsigned int) * blockDimX32 * blockDimY32 * 1024);
	memset (res_mask, 0 , sizeof(unsigned char) * blockDimX32 * (blockDimY32 << 10));
	
	/*
		п.1. Восстанавливаем истинное значение маски на основе res_cpu и plan_old (т.е. значений массива plan до минимизации)
	*/
	/*		// Используется для корректной проверки DataToQueueB
	
		for (int i=0; i<blockDimY32;i++ )
		for (int j=0; j<blockDimX32;j++ )
			for (int k=0;k<32;k++)
				for (int l=0;l<32;l++)	
				{   int offset  = (k*32 + l);
					int _x = (plan_old[1024*(i*blockDimX32+j) + offset])&0x1F;
					int _y = (plan_old[1024*(i*blockDimX32+j) + offset])>>5;
					if ( res_cpu[1024*(i*blockDimX32+j) + offset]>0 && plan_old[1024*(i*blockDimX32+j) + offset]<=1024)
					{
							if (i*32 +_y <windows_height && j*32 + _x <windows_width)
								dump_full_mask_cpu[i*windows_width*32+j*32+_y*windows_width+_x] =  255;
					}
				}*/
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

	for (int i = 0; i < windows_height; i++)
		// чистим правый крайний столбец маски
		res_mask[windows_width * (i + 1) - 1] = 0;

	free(_integral_img);
	free(varince_result);
	free(res_cpu );
	//free(plan_old );
	//free(result_cpu );
	free(plans_cpu );
	//free(dump_plans_cpu );
	//free(dump_full_mask_cpu);

#ifdef DEBUG_MODE
	char str[256];
	sprintf(str,"%s_full_mask.pgm",res_fname);
	FILE *fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgmt(fp,windows_width, windows_height, res_mask);
		fclose(fp);
	}
#endif
	
	/*for (int i=0; i<blockDimY32;i++ )
		for (int j=0; j<blockDimX32;j++ )
			for (int k=0;k<32;k++)
				for (int l=0;l<32;l++)	
				{   
						int offset  = (k*32 + l);
						dump_plans_cpu[i*windows_width32*32+j*32+k*windows_width32+l] = plans_cpu[1024*(i*blockDimX32+j) + offset];
				}
	printf("Finishing\n");

	sprintf(str,"%s_plan.pgm",res_fname);
	fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm4t(fp,windows_width32, windows_height32, dump_plans_cpu);
		fclose(fp);
	}
	
	unsigned int *bbcnt = (unsigned int *)malloc(sizeof(unsigned int) * blockDimX32 * blockDimY32);
	memset(dump_plans_cpu,0, sizeof(unsigned int) * blockDimX32 * blockDimY32 * 1024);
	for (int i=0; i<blockDimY32;i++ )
		for (int j=0; j<blockDimX32;j++ )
		{   
			int m = 0, n = 0;
			for (int k=0;k<32;k++)
				for (int l=0;l<32;l++)	
				{
					int offset = k*32 + l;
					if (i*32+k < windows_height && j*32 + l < windows_width && dump_full_mask_cpu[i*windows_width*32+j*32+k*windows_width+l] >0)
					{
						dump_plans_cpu[i*windows_width32*32+j*32+m*windows_width32+n] = offset;
						n++;
						if (n == 32)
						{ n = 0; m++;}
					}
				}
			bbcnt[blockDimX32*i+j] = m*32 + n;	
		}
	sprintf(str,"%s_plan_etalon.pgm",res_fname);
	// сохраняем итоговое значение маски
	fp = fopen(str,"wb");
	if (fp!=0)
	{
		save_pgm4t(fp,windows_width32, windows_height32, dump_plans_cpu);
		fclose(fp);
	}
	
	sprintf(str,"%s_plan_cnt.pgm",res_fname);
	fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm4t(fp,blockDimX32, blockDimY32, bbcnt);
		fclose(fp);
	}
	
	/*char str[256];
	sprintf(str,"%s.pgm",res_fname);
	FILE *fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm(fp,windows_width, windows_height, res_mask);
		fclose(fp);
	}
	sprintf(str,"%s_plan.pgm",res_fname);
	fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm4t(fp,windows_width32, windows_height32, dump_plans_cpu);
		fclose(fp);
	}
	
	unsigned int *bbcnt = (unsigned int *)malloc(sizeof(unsigned int) * blockDimX32 * blockDimY32);
	
	memset(dump_plans_cpu,0, sizeof(unsigned int) * blockDimX32 * blockDimY32 * 1024);
	
	 for (int i=0; i<blockDimY32;i++ )
		for (int j=0; j<blockDimX32;j++ )
		{   
			int m = 0, n = 0;
			for (int k=0;k<32;k++)
				for (int l=0;l<32;l++)	
				{
					if (i*32+k < windows_height && j*32 + l < windows_width && res_mask[i*windows_width*32+j*32+k*windows_width+l] >0)
					{
						int offset = k*32 + l;
						dump_plans_cpu[i*windows_width32*32+j*32+m*windows_width32+n] = offset;
						n++;
						if (n == 32)
						{ n = 0; m++;}
					}
				}
				
			bbcnt[blockDimX32*i+j] = m*32 + n;
		}
	sprintf(str,"%s_plan_cpu.pgm",res_fname);
	fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm4t(fp,windows_width32, windows_height32, dump_plans_cpu);
		fclose(fp);
	}
	
	sprintf(str,"%s_plan_cnt.pgm",res_fname);
	fp = fopen(str,"wb");
	if (fp!=0)		
	{
		save_pgm4t(fp,blockDimX32, blockDimY32, bbcnt);
		fclose(fp);
	}*/
}
