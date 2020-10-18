#include "cudaGenerator.h"

#include <stdio.h>
#include <cxtypes.h>
#include <cvtypes.h>

// ф-ция аналогична SetCascadeToGPU. Собираем внутренние структуры, которые в дальнейшем будем обрабатывать
int cudaGenerator::getCascades(void* _cascade_, int first_cascade, int cascadeCount, int gpu_window_width)
{
	CvHaarClassifierCascade *cascade = (CvHaarClassifierCascade *)_cascade_;
	CvHaarStageClassifier *stages = &cascade->stage_classifier[first_cascade];
	int node2_count = 0;
	int node3_count = 0;
	int _error = 0;

#ifdef _OPENMP
	#pragma omp parallel for shared(stages, _error) reduction(+: node2_count, node3_count)
#endif
	for (int i=0; i<cascadeCount;i++)
	{
		CvHaarClassifier* classifier = stages[i].classifier;
		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier->count != 1)
				_error = 1;
			if (classifier[j].haar_feature->rect[2].weight==0)
				node2_count++;
			else
				node3_count++;
		}
	}
	if (_error > 0)
		return -1;
	
	__gpu_rect2s = (_gpu_rect*)malloc(sizeof(_gpu_rect)*(node2_count<<1));
	__gpu_rect3s = (_gpu_rect*)malloc(sizeof(_gpu_rect)*3*node3_count);
	__haar_rect2_weights = (float*)malloc(sizeof(float)*(node2_count<<1));
	__haar_rect3_weights = (float*)malloc(sizeof(float)*node3_count*3);

	__haar_nodes = (_gpu_node*)malloc(sizeof(_gpu_node)*(node2_count + node3_count));
	__haar_cascade = (_gpu_cascade *)malloc(sizeof(_gpu_cascade)*cascadeCount);

	const float icv_stage_threshold_bias = 0.0001f;
	gpu_window_width = cascade->orig_window_size.width + gpu_window_width;  //16+20 || 32+20

	int node2_pos = 0, node3_pos = 0, node_pos = 0;
	for (int i=0; i<cascadeCount; i++)
	{
		int node2_count = 0, node3_count = 0;
		__haar_cascade[i].threashold = stages[i].threshold - icv_stage_threshold_bias;
		__haar_cascade[i].node2_first = node2_pos;
		__haar_cascade[i].node3_first = node3_pos;
		__haar_cascade[i].node_position = node_pos;
		CvHaarClassifier* classifier = stages[i].classifier;

		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier[j].haar_feature->rect[2].weight == 0)
			{
				CvRect *curr_rect = &classifier[j].haar_feature->rect[0].r;
				int x = curr_rect->x;
				int y = curr_rect->y;
				__gpu_rect2s[node2_pos].p1 = (y * gpu_window_width + x)<<2;
				__gpu_rect2s[node2_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				__gpu_rect2s[node2_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				__gpu_rect2s[node2_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				__haar_rect2_weights[node2_pos] = (float)(classifier[j].haar_feature->rect[0].weight * _isq);

				curr_rect = &classifier[j].haar_feature->rect[1].r;
				node2_pos++;
				x = curr_rect->x;
				y = curr_rect->y;
				__gpu_rect2s[node2_pos].p1 = (y * gpu_window_width + x)<<2;
				__gpu_rect2s[node2_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				__gpu_rect2s[node2_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				__gpu_rect2s[node2_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				__haar_rect2_weights[node2_pos] = (float)(classifier[j].haar_feature->rect[1].weight * _isq);
				node2_pos++;
				node2_count++;

				__haar_nodes[node_pos].threashold = classifier[j].threshold[0];
				__haar_nodes[node_pos].a = classifier[j].alpha[0];
				__haar_nodes[node_pos].b = classifier[j].alpha[1];
				node_pos++;
			}
			else
			{
				CvRect *curr_rect = &classifier[j].haar_feature->rect[0].r;
				int x = curr_rect->x;
				int y = curr_rect->y;
				__gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				__gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				__haar_rect3_weights[node3_pos] = (float)(classifier[j].haar_feature->rect[0].weight * _isq);
				node3_pos++;
				curr_rect = &classifier[j].haar_feature->rect[1].r;
				x = curr_rect->x;
				y = curr_rect->y;
				__gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				__gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				__haar_rect3_weights[node3_pos] = (float)(classifier[j].haar_feature->rect[1].weight * _isq);
				node3_pos++;  
				curr_rect = &classifier[j].haar_feature->rect[2].r;
				x = curr_rect->x;
				y = curr_rect->y;
				__gpu_rect3s[node3_pos].p1 = (y * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p2 = (y * gpu_window_width + x + curr_rect->width)<<2;
				__gpu_rect3s[node3_pos].p3 = ((y + curr_rect->height) * gpu_window_width + x)<<2;
				__gpu_rect3s[node3_pos].p4 = ((y + curr_rect->height) * gpu_window_width + x + curr_rect->width)<<2;
				__haar_rect3_weights[node3_pos] = (float)(classifier[j].haar_feature->rect[2].weight * _isq);
				node3_pos++;  
				node3_count++;
			}
		}

		for (int j=0;j<stages[i].count;j++)
		{
			if (classifier[j].haar_feature->rect[2].weight != 0)
			{
				__haar_nodes[node_pos].threashold = classifier[j].threshold[0];
				__haar_nodes[node_pos].a = classifier[j].alpha[0];
				__haar_nodes[node_pos].b = classifier[j].alpha[1];
				node_pos++;
			}
		}

		__haar_cascade[i].node2_count = node2_count;
		__haar_cascade[i].node3_count = node3_count;
	}
	return 1;
}

// составляем заголовок для haar_first_stage
void cudaGenerator::generateInitialization_FirstStage(FILE *fp)
{
	fprintf(fp, "extern \"C\" __global__ static void haar_first_stage_1(unsigned int *img ,float* sqsum, unsigned int *res, \n");
	fprintf(fp, "									unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,\n");
	fprintf(fp, "									unsigned int sqsum_buffer_width, unsigned int cascade_count)\n");
	fprintf(fp, "{\n");
	fprintf(fp, "	__shared__ unsigned int _img[36][36];\n");
	fprintf(fp, "	__shared__ float _sqsum[16][16];\n");
	fprintf(fp, "	__shared__ unsigned char _res[256];\n");
	fprintf(fp, "	int _imgX = threadIdx.x;\n");
	fprintf(fp, "	int _imgY = threadIdx.y;\n");
	fprintf(fp, "	unsigned int PosX = (blockIdx.x<<4) + threadIdx.x;\n");
	fprintf(fp, "	unsigned int PosY = (blockIdx.y<<4) + threadIdx.y;\n");
	fprintf(fp, "\n	int img_pos = __umul24(PosY, img_buffer_width) + PosX;\n");
	fprintf(fp, "\n	_img[_imgY][_imgX] = img[img_pos];\n");
	fprintf(fp, "	_img[_imgY][_imgX+16] = img[img_pos+16];\n");
	fprintf(fp, "	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];\n");
	fprintf(fp, "	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];\n");
	fprintf(fp, "	if (threadIdx.x<4 && PosX+32<img_buffer_width )\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY][_imgX+32] = img[img_pos + 32];\n");
	fprintf(fp, "		_img[_imgY+16][_imgX+32] = img[img_pos+32 + (img_buffer_width<<4)];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	if (threadIdx.y<4 && PosY+32<img_height)\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];\n");
	fprintf(fp, "		_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5)+16];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	if (threadIdx.x<4 && threadIdx.y<4 && PosX+32<img_buffer_width && PosY+32<img_height )\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5)+32];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	int sqsum_pos = __umul24(PosY, sqsum_buffer_width) + PosX;\n");
	fprintf(fp, "	_sqsum[_imgY][_imgX] = sqsum[sqsum_pos];\n");
	fprintf(fp, "	int res_pos = (threadIdx.y<<4) + threadIdx.x;\n");
	fprintf(fp, "	_res[res_pos] = 255;\n");
	fprintf(fp, "	__syncthreads();\n");
	fprintf(fp, "	if ((PosX + window_STEPX + 3<img_width) && (PosY + window_STEPY + 3<img_height))\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		unsigned char* cur_img = (unsigned char*)&_img[threadIdx.y][threadIdx.x];\n");
	fprintf(fp, "		float variance_norm_factor = _sqsum[_imgY][_imgX];\n");
}

// составляем окончание для haar_first_stage
void cudaGenerator::generateFinalization_FirstStage(FILE *fp)
{
	fprintf(fp, "	}\n");
	fprintf(fp, "	__syncthreads();\n");
	fprintf(fp, "	unsigned int *_resI = (unsigned int *)_res;\n");
	fprintf(fp, "	if (threadIdx.y<4)\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "			int resPos =  ( (__umul24(gridDim.x, blockIdx.y) + blockIdx.x)<<6 ) + res_pos;\n");
	fprintf(fp, "			res[resPos] = _resI[res_pos];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "}\n");
}

// генерируем промежуточную часть для haar_first_stage
void cudaGenerator::generateMiddle_FirstStage(int cascadeCount, FILE *fp)
{
	_gpu_node* curr_node = __haar_nodes;
	for (int k = 0; k < cascadeCount; k++)
	{
		if (k == 0)
			fprintf(fp, "		float stage_sum = 0.0f;\n");
		else
		{
			fprintf(fp, "		if (_res[res_pos] == 255)\n");
			fprintf(fp, "		{\n");
			fprintf(fp, "			stage_sum = 0.0f;\n\n");
		}
		int j = __haar_cascade[k].node2_first;
		_gpu_rect *curr_rec = &__gpu_rect2s[j];
		fprintf(fp, "//		>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
		for (unsigned int i=0; i<__haar_cascade[k].node2_count; i++ )
		{
			if (i == 0 && k == 0)
			{
				fprintf(fp, "		float t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
				fprintf(fp, "		int sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "		float r = (sum * %.10ff);\n", __haar_rect2_weights[j]);
			}
			else
			{
				fprintf(fp, "		t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
				fprintf(fp, "		sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "		r = (sum * %.10ff);\n", __haar_rect2_weights[j]);
			}
			curr_rec++;
			j++;
			fprintf(fp, "		sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
			fprintf(fp, "		r += (sum * %.10ff);\n", __haar_rect2_weights[j]);
			fprintf(fp, "		stage_sum += r < t ? %.10ff : %.10ff;\n", curr_node->a, curr_node->b);
			fprintf(fp, "//		>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
			curr_rec++;
			j++;
			curr_node++;
		}
		j = __haar_cascade[k].node3_first;
		curr_rec = &__gpu_rect3s[j];
		for (unsigned int i=0; i<__haar_cascade[k].node3_count;i++ )
		{
			fprintf(fp, "		t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
			fprintf(fp, "		sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
			fprintf(fp, "		r = (sum * %.10ff);\n", __haar_rect3_weights[j]);
			curr_rec++;
			j++;
			fprintf(fp, "		sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
			fprintf(fp, "		r += (sum * %.10ff);\n", __haar_rect3_weights[j]);
			curr_rec++;
			j++;
			fprintf(fp, "		sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
							curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
			fprintf(fp, "		r += (sum * %.10ff);\n", __haar_rect3_weights[j]);
			fprintf(fp, "		stage_sum += r < t ? %.10ff : %.10ff;\n", curr_node->a, curr_node->b);
			fprintf(fp, "//		>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
			curr_rec++;
			j++;
			curr_node++;
		}
		fprintf(fp, "		if( stage_sum < %.10ff )\n", __haar_cascade[k].threashold);
		fprintf(fp, "			_res[res_pos] = 0;\n");
		fprintf(fp, "//		------------------------------------------------------------------------\n");
		if (k > 0)
			fprintf(fp, "		}\n\n");
	}
}

void cudaGenerator::generateInitialization(int index, int mode, int indexElement, FILE *fp)
{
	fprintf(fp, "extern \"C\" __global__ static void haar_next_stage_%d_%d(unsigned int *img ,float* sqsum, unsigned int *res,\n", indexElement, index);
	fprintf(fp, "								   unsigned int img_width , unsigned int img_height, unsigned int img_buffer_width,\n");
	fprintf(fp, "								   unsigned int sqsum_buffer_width, unsigned int cascade_count, unsigned int *plan,\n");
	fprintf(fp, "								   unsigned short *elementIndex)\n");
	fprintf(fp, "{\n");
	fprintf(fp, "	__shared__ unsigned int _img[52][52];\n");
	fprintf(fp, "	__shared__ float _sqsum[32*32];\n");
	fprintf(fp, "	__shared__ unsigned char _res[256];\n");
	fprintf(fp, "	__shared__ unsigned short _plan[256];\n");
	fprintf(fp, "	__shared__ float _stage_sum[128];\n");
	fprintf(fp, "	__shared__ unsigned short elementsIndex;\n");
	fprintf(fp, "	int _imgX = threadIdx.x;\n");
	fprintf(fp, "	int _imgY = threadIdx.y;\n");
	fprintf(fp, "	unsigned int PosX = (blockIdx.x<<5) + threadIdx.x;\n");
	fprintf(fp, "	unsigned int PosY = (blockIdx.y<<5) + threadIdx.y;\n");
	fprintf(fp, "	int img_pos = __umul24(PosY, img_buffer_width) + PosX;\n");
	fprintf(fp, "	_img[_imgY][_imgX] = img[img_pos];\n");
	fprintf(fp, "	_img[_imgY][_imgX+16] = img[img_pos+16];\n");
	fprintf(fp, "	_img[_imgY+16][_imgX] = img[img_pos+(img_buffer_width<<4)];\n");
	fprintf(fp, "	_img[_imgY+16][_imgX+16] = img[img_pos+(img_buffer_width<<4) + 16];\n");
	fprintf(fp, "	_img[_imgY][_imgX+32] = img[img_pos+32];\n");
	fprintf(fp, "	_img[_imgY+16][_imgX+32] = img[img_pos+32+(img_buffer_width<<4)];\n");
	fprintf(fp, "	_img[_imgY+32][_imgX] = img[img_pos+(img_buffer_width<<5)];\n");
	fprintf(fp, "	_img[_imgY+32][_imgX+16] = img[img_pos+(img_buffer_width<<5) + 16];\n");
	fprintf(fp, "	_img[_imgY+32][_imgX+32] = img[img_pos+(img_buffer_width<<5) + 32];\n");
	fprintf(fp, "	if (threadIdx.x<4 && PosX+48<img_buffer_width )\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY][_imgX+48] = img[img_pos + 48];\n");
	fprintf(fp, "		_img[_imgY+16][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<4)];\n");
	fprintf(fp, "		_img[_imgY+32][_imgX+48] = img[img_pos + 48 + (img_buffer_width<<5)];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	if (threadIdx.y<4 && PosY+48<img_height)\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY+48][_imgX] = img[img_pos+__umul24(img_buffer_width, 48)];\n");
	fprintf(fp, "		_img[_imgY+48][_imgX+16] = img[img_pos+__umul24(img_buffer_width, 48)+16];\n");
	fprintf(fp, "		_img[_imgY+48][_imgX+32] = img[img_pos+__umul24(img_buffer_width, 48)+32];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	if (threadIdx.x<4 && threadIdx.y<4 && PosX+48<img_buffer_width && PosY+48<img_height )\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		_img[_imgY+48][_imgX+48] = img[img_pos+__umul24(img_buffer_width, 48)+48];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	int sqsum_pos = __umul24(PosY, sqsum_buffer_width) + PosX;\n");
	fprintf(fp, "	_sqsum[(_imgY<<5) + _imgX] = sqsum[sqsum_pos];\n");
	fprintf(fp, "	_sqsum[(_imgY<<5) + _imgX+16] = sqsum[sqsum_pos + 16];\n");
	fprintf(fp, "	_sqsum[((_imgY+16)<<5) + _imgX] = sqsum[sqsum_pos + (sqsum_buffer_width<<4)];\n");
	fprintf(fp, "	_sqsum[((_imgY+16)<<5) + _imgX+16] = sqsum[sqsum_pos + (sqsum_buffer_width<<4) + 16];\n");
	fprintf(fp, "	int lpos = (threadIdx.y<<4) + threadIdx.x;\n");
	fprintf(fp, "	int _base_offset = __umul24(blockIdx.y, gridDim.x)+blockIdx.x;\n");
	fprintf(fp, "	int basic_plan_pos = _base_offset<<9;\n");
	fprintf(fp, "	unsigned int saveOffset = (_base_offset << 8) + lpos;\n");
	fprintf(fp, "	res[ saveOffset ] = 0;\n");
	fprintf(fp, "	if (lpos == 0 )\n");
	fprintf(fp, "		elementsIndex = elementIndex[_base_offset];\n");

	if (mode == 2)
	{
		// версия на 2 потока
		fprintf(fp, "	unsigned int plan_pos = (lpos & 31) + ( ( (lpos & 192) >> 6 ) << 5);\n");
		fprintf(fp, "	unsigned int arrayPos = ( lpos & 32 ) >> 5;\n");
	}
	if (mode == 4)
	{
		// версия на 4 потока
		fprintf(fp, "	unsigned int plan_pos = (lpos & 31) + (((lpos & 128) >> 7)<<5);\n");
		fprintf(fp, "	unsigned int arrayPos = (lpos & 96) >> 5;\n");
		fprintf(fp, "	int writePos = ( ( (arrayPos & 2) >> 1 ) <<6 ) + plan_pos;\n");
	}
	if (mode == 8)
	{
		fprintf(fp, "	unsigned int plan_pos = lpos & 31;\n");
		fprintf(fp, "	unsigned int arrayPos = (lpos & 224) >> 5;\n");
		fprintf(fp, "	int writePos = ( ( (arrayPos & 6) >> 1 )<<5 ) + plan_pos;\n");
	}
	fprintf(fp, "	__syncthreads();\n");

	fprintf(fp, "	unsigned short elemCount = elementsIndex;\n");
	fprintf(fp, "	int baseOffset = 0;\n");
	fprintf(fp, "	int flag = 0;\n");

	if( mode == 2 )
	{
		fprintf(fp, "	for (int basic_plan_offset = 0; basic_plan_offset<elemCount; basic_plan_offset+=64)\n");
		fprintf(fp, "	{\n");
		fprintf(fp, "		if ( ( basic_plan_offset & 64 ) == 0 )\n");
		fprintf(fp, "			_res[lpos] = 0;\n");
		fprintf(fp, "		if (lpos<128 && ( basic_plan_offset & 64 ) == 0)\n");
	}
	else
	{
		if( mode == 4 )
			fprintf(fp, "	for (int basic_plan_offset = 0; basic_plan_offset<elemCount; basic_plan_offset+=32)\n");
		if( mode == 8 )
			fprintf(fp, "	for (int basic_plan_offset = 0; basic_plan_offset<elemCount; basic_plan_offset+=16)\n");
		fprintf(fp, "	{\n");
		fprintf(fp, "		if ( ( basic_plan_offset & 127 ) == 0 )\n");
		fprintf(fp, "			_res[lpos] = 0;\n");
		fprintf(fp, "		if (lpos<128 && ( basic_plan_offset & 127 ) == 0)\n");
	}
	fprintf(fp,	"		{\n");
	fprintf(fp, "			unsigned int *_planI = (unsigned int*)_plan;\n");
	fprintf(fp, "			_planI[lpos] = plan[basic_plan_pos + basic_plan_offset + lpos];\n");
	fprintf(fp, "			baseOffset = basic_plan_offset;\n");
	fprintf(fp, "			flag = 1;\n");
	fprintf(fp, "		}\n");
	fprintf(fp, "		__syncthreads();\n");
	if( mode == 2 )
		fprintf(fp, "		int offset = ( basic_plan_offset & 64 ) << 1;\n");
	if( mode == 4 )
		fprintf(fp, "		int offset = ( basic_plan_offset & 0x60 ) << 1;\n");
	if( mode == 8 )
		fprintf(fp, "		int offset = ( basic_plan_offset & 0x70 ) << 1;\n");
}

void cudaGenerator::generateFinalization(int mode, FILE *fp)
{
	fprintf(fp, "		unsigned int *_resI = (unsigned int *)_res;\n");
	if (mode == 2)
		fprintf(fp, "		if (threadIdx.y<4 && offset == 128)\n");
	if (mode == 4)
		fprintf(fp, "		if (threadIdx.y<4 && offset == 192)\n");
	if (mode == 8)
		fprintf(fp, "		if (threadIdx.y<4 && offset == 224)\n");
	fprintf(fp, "		{\n");
	fprintf(fp, "			int resPos =  saveOffset + (baseOffset>>1);\n");
	fprintf(fp, "			res[resPos] = _resI[lpos];\n");
	fprintf(fp, "			flag = 0;\n");
	fprintf(fp, "		}\n");
	fprintf(fp, "		__syncthreads();\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "	unsigned int *_resI = (unsigned int *)_res;\n");
	fprintf(fp, "	if (lpos < 64 && flag == 1)\n");
	fprintf(fp, "	{\n");
	fprintf(fp, "		int resPos =  saveOffset + (baseOffset>>1);\n");
	fprintf(fp, "		res[resPos] = _resI[lpos];\n");
	fprintf(fp, "	}\n");
	fprintf(fp, "}\n");
}
void cudaGenerator::generateMiddle(int base_position, int mode, int cascadeCount, FILE *fp)
{
	fprintf(fp, "		unsigned short plan_item = _plan[offset + plan_pos];\n"/*, res_base_pos*/);
	fprintf(fp, "		float stage_sum = 0.0f;\n");

	int length = 2, length2 = 2;
	int curr_rec_inc = 3, curr_rec_inc2 = 4;
	if (mode != 2)
	{
		if (mode == 4)	length = 4;
		else			length = 8;
		length2 = 4;
		curr_rec_inc = 7;
		curr_rec_inc2 = 10;
	}

	for (int cascadeElem = 0; cascadeElem < cascadeCount; cascadeElem++)
	{
		_gpu_cascade* curr_cascade = &__haar_cascade[base_position + cascadeElem];
		if ( cascadeElem != 0 )
		{
			fprintf(fp, "		if (_res[offset + plan_pos] == 255)\n");
			fprintf(fp, "		{\n");
			fprintf(fp, "			stage_sum = 0.0f;\n");
		}
		for (int arrayPos = 0; arrayPos < length; arrayPos++)
		{
			int haar_start_position = curr_cascade->node_position + arrayPos;
			fprintf(fp, "		if (plan_item <= 1024 && arrayPos == %d)\n", arrayPos);
			fprintf(fp, "		{\n");
			fprintf(fp, "			unsigned char* cur_img = (unsigned char*)&_img[(plan_item>>5)][plan_item&0x1F];\n");
			fprintf(fp, "			float variance_norm_factor = _sqsum[plan_item];\n");
			_gpu_node* curr_node = &__haar_nodes[haar_start_position];
			int j = curr_cascade->node2_first + (arrayPos<<1);
			_gpu_rect *curr_rec = &__gpu_rect2s[j];

			for (unsigned int i = arrayPos; i<curr_cascade->node2_count; i+=length2 )
			{
				if (i == arrayPos)
				{
					fprintf(fp, "			float t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
					fprintf(fp, "			int sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
														curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
					fprintf(fp, "			float r = %.10ff * sum;\n", __haar_rect2_weights[j]);
				}
				else
				{
					fprintf(fp, "			t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
					fprintf(fp, "			sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
													curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
					fprintf(fp, "			r = %.10ff * sum;\n", __haar_rect2_weights[j]);
				}
				curr_rec++;
				j++;
				fprintf(fp, "			sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
													curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "			r += (%.10ff * sum);\n", __haar_rect2_weights[j]);
				fprintf(fp, "			stage_sum += r < t ? %.10ff : %.10ff;\n", curr_node->a, curr_node->b);
				curr_rec += curr_rec_inc;
				j += curr_rec_inc;
				curr_node+=length2;
				fprintf(fp, "//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
			}

			fprintf(fp, "//-------------------------------------------------------------\n");

			curr_node = &__haar_nodes[curr_cascade->node2_count + haar_start_position];
			j = curr_cascade->node3_first + arrayPos*3;
			curr_rec = &__gpu_rect3s[j];

			for (unsigned int i=arrayPos; i<curr_cascade->node3_count;i+=length2 )
			{
				fprintf(fp, "			t = %.10ff * variance_norm_factor;\n", curr_node->threashold);
				fprintf(fp, "			sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
												curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "			r = %.10ff * sum;\n", __haar_rect3_weights[j]);
				curr_rec++;
				j++;
				fprintf(fp, "			sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
												curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "			r += (%.10ff * sum);\n", __haar_rect3_weights[j]);
				curr_rec++;
				j++;
				fprintf(fp, "			sum =  *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) - *(unsigned int*)(cur_img + %d) + *(unsigned int*)(cur_img + %d);\n",
												curr_rec->p1, curr_rec->p2, curr_rec->p3, curr_rec->p4);
				fprintf(fp, "			r += (%.10ff * sum);\n", __haar_rect3_weights[j]);
				fprintf(fp, "			stage_sum += r < t ? %.10ff : %.10ff;\n", curr_node->a, curr_node->b);
				curr_rec+=curr_rec_inc2;
				j+=curr_rec_inc2;
				curr_node+=length2;
				fprintf(fp, "//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
			}
			fprintf(fp, "//-------------------------------------------------------------\n");
			if (arrayPos == 1 && mode == 2)
				fprintf(fp, "			_stage_sum[plan_pos] = stage_sum;\n");
			if ( ( arrayPos & 1 ) == 1 && ( mode == 4 || mode == 8 ) )
				fprintf(fp, "			_stage_sum[writePos] = stage_sum;\n");
			fprintf(fp, "		}\n");
		}
		if ( cascadeElem != 0 )
			fprintf(fp, "		}\n");
		fprintf(fp, "		__syncthreads();\n");

		if (mode == 2)
		{
			if ( cascadeElem != 0 )
				fprintf(fp, "		if (_res[offset + plan_pos] == 255 && arrayPos == 0 && plan_item <= 1024)\n");
			else
				fprintf(fp, "		if (arrayPos == 0 && plan_item <= 1024)\n");
			fprintf(fp, "		{\n");
			if ( cascadeElem != 0 )
				fprintf(fp, "			_res[offset + plan_pos] = 0;\n");
			fprintf(fp, "				stage_sum = _stage_sum[plan_pos] + stage_sum;\n");
			fprintf(fp, "				if( stage_sum > %.10ff )\n", curr_cascade->threashold);
			fprintf(fp, "					_res[offset + plan_pos] = 255;\n");
			fprintf(fp, "		}\n");
		}
		if (mode == 4)
		{
			if ( cascadeElem != 0 )
				fprintf(fp, "		if (_res[offset + plan_pos] == 255 && plan_item <= 1024 && ( arrayPos & 1) == 0)\n");
			else
				fprintf(fp, "		if (plan_item <= 1024 && ( arrayPos & 1) == 0)\n");
			fprintf(fp, "		{\n");
			fprintf(fp, "			stage_sum += _stage_sum [writePos];\n");
			fprintf(fp, "			if (arrayPos == 2)\n");
			fprintf(fp, "				_stage_sum [plan_pos] = stage_sum;\n");
			fprintf(fp, "		}\n");
			fprintf(fp, "		__syncthreads();\n");
			if ( cascadeElem != 0 )
				fprintf(fp, "		if (_res[offset + plan_pos] == 255 && plan_item <= 1024 && arrayPos == 0)\n");
			else
				fprintf(fp, "		if (plan_item <= 1024 && arrayPos == 0)\n");
			fprintf(fp, "		{\n");
			if ( cascadeElem != 0 )
				fprintf(fp, "			_res[offset + plan_pos] = 0;\n");
			fprintf(fp, "			stage_sum = _stage_sum[plan_pos] + stage_sum;\n");
			fprintf(fp, "			if( stage_sum > %.10ff )\n", curr_cascade->threashold);
			fprintf(fp, "				_res[offset + plan_pos] = 255;\n"/*, res_base_pos*/);
			fprintf(fp, "		}\n");
		}
		if (mode == 8)
		{
			if ( cascadeElem != 0 )
				fprintf(fp, "		if (_res[offset + plan_pos] == 255 && (arrayPos & 1) == 0 && plan_item <= 1024)\n");
			else
				fprintf(fp, "		if ( (arrayPos & 1) == 0 && plan_item <= 1024)\n");
			fprintf(fp, "		{\n");
			fprintf(fp, "			int pos = (arrayPos & 7) >> 1;\n");
			fprintf(fp, "			stage_sum += _stage_sum [writePos];\n");
			fprintf(fp, "			if ( pos > 0)\n");
			fprintf(fp, "			_stage_sum [ ( (pos - 1) << 5 ) + plan_pos] = stage_sum;\n");
			fprintf(fp, "		}\n");
			fprintf(fp, "		__syncthreads();\n");
			if ( cascadeElem != 0 )
				fprintf(fp, "		if (_res[offset + plan_pos] == 255 && arrayPos == 0 && plan_item <= 1024)\n");
			else
				fprintf(fp, "		if ( arrayPos == 0 && plan_item <= 1024)\n");
			fprintf(fp, "		{\n");
			if ( cascadeElem != 0 )
				fprintf(fp, "			_res[offset + plan_pos] = 0;\n");
			fprintf(fp, "			stage_sum = _stage_sum[plan_pos] + _stage_sum[32 + plan_pos] + _stage_sum[64 + plan_pos] + stage_sum;\n");
			fprintf(fp, "			if( stage_sum > %.10ff )\n", curr_cascade->threashold);
			fprintf(fp, "				_res[offset + plan_pos] = 255;\n");
			fprintf(fp, "		}\n");
		}
		fprintf(fp, "		__syncthreads();\n");
		fprintf(fp, "//*************************************************************\n");
	}
}

void cudaGenerator::generateHeader(void *_kernelFile)
{
	FILE *kernelFile = (FILE *)_kernelFile;
	fprintf(kernelFile, "#pragma once\n");
	fprintf(kernelFile, "#define window_STEPX 18\n");
	fprintf(kernelFile, "#define window_STEPY 18\n");
	fprintf(kernelFile, "#define PROC_BLOCK_SIZE 1024\n");
	fprintf(kernelFile, "#define PROC_BLOCK_SIZE_LOG 10\n");
	fprintf(kernelFile, "#define PROC_BLOCK_SIZE4 256\n");
	fprintf(kernelFile, "#define PROC_BLOCK_SIZE4_LOG 8\n");
	fprintf(kernelFile, "#define THREADS 256\n");
	fprintf(kernelFile, "#define LOG_NUM_BANKS_SHORT 5\n");
	fprintf(kernelFile, "#define CONFLICT_FREE_OFFSET(index) (((index) >> LOG_NUM_BANKS_SHORT)<<1)\n");

	fprintf(kernelFile, "#define getOffset() {								\\\n");
	fprintf(kernelFile, "	ai = offset * ( ( thid << 1 ) + 1 ) - 1;		\\\n");
	fprintf(kernelFile, "	bi = offset * ( ( thid << 1 ) + 2 ) - 1;		\\\n");
	fprintf(kernelFile, "	ai += CONFLICT_FREE_OFFSET(ai);					\\\n");
	fprintf(kernelFile, "	bi += CONFLICT_FREE_OFFSET(bi);					\\\n");
	fprintf(kernelFile, "}\n");

	fprintf(kernelFile, "#define condition_1(value) {						\\\n");
	fprintf(kernelFile, "	__syncthreads();								\\\n");
	fprintf(kernelFile, "	if (thid < value)	  							\\\n");
	fprintf(kernelFile, "	{												\\\n");
	fprintf(kernelFile, "		getOffset();								\\\n");
	fprintf(kernelFile, "		sum[bi] += sum[ai];							\\\n");
	fprintf(kernelFile, "	}												\\\n");
	fprintf(kernelFile, "	offset = offset << 1;							\\\n");
	fprintf(kernelFile, "}\n");

	fprintf(kernelFile, "#define condition_2(value) {						\\\n");
	fprintf(kernelFile, "	offset = offset>>1;								\\\n");
	fprintf(kernelFile, "	__syncthreads();								\\\n");
	fprintf(kernelFile, "	if (thid < value)								\\\n");
	fprintf(kernelFile, "	{												\\\n");
	fprintf(kernelFile, "		getOffset();								\\\n");
	fprintf(kernelFile, "		unsigned short t = sum[ai];					\\\n");
	fprintf(kernelFile, "		sum[ai] = sum[bi];							\\\n");
	fprintf(kernelFile, "		sum[bi] += t;								\\\n");
	fprintf(kernelFile, "	}												\\\n");
	fprintf(kernelFile, "}\n");

	fprintf(kernelFile, "__device__  unsigned short scan(unsigned short *sum, unsigned short *len )\n");
	fprintf(kernelFile, "{\n");
	fprintf(kernelFile, "	int offset = 1;\n");
	fprintf(kernelFile, "	int thid  = threadIdx.x;\n");
	fprintf(kernelFile, "	int ai = offset * ( ( thid << 1 ) + 1 ) - 1;\n");
	fprintf(kernelFile, "	int bi = offset * ( ( thid << 1 ) + 2 ) - 1;\n");
	fprintf(kernelFile, "	ai += CONFLICT_FREE_OFFSET(ai);\n");
	fprintf(kernelFile, "	bi += CONFLICT_FREE_OFFSET(bi);\n");
	fprintf(kernelFile, "	sum[bi] += sum[ai];\n");
	fprintf(kernelFile, "	offset = offset << 1;\n");
		
	fprintf(kernelFile, "	condition_1( 128 );\n");
	fprintf(kernelFile, "	condition_1(  64 );\n");
	fprintf(kernelFile, "	condition_1(  32 );\n");
	fprintf(kernelFile, "	condition_1(  16 );\n");
	fprintf(kernelFile, "	condition_1(   8 );\n");
	fprintf(kernelFile, "	condition_1(   4 );\n");
	fprintf(kernelFile, "	condition_1(   2 );\n");
	fprintf(kernelFile, "	condition_1(   1 );\n");

	fprintf(kernelFile, "	if (thid == 0)\n");
	fprintf(kernelFile, "	{\n");
	fprintf(kernelFile, "		int index = 2*THREADS - 1;\n");
	fprintf(kernelFile, "		index += CONFLICT_FREE_OFFSET(index);\n");
	fprintf(kernelFile, "		*len = ( sum[index] >> 1 );\n");
	fprintf(kernelFile, "		sum[index] = 0;\n");
	fprintf(kernelFile, "	}\n");

	fprintf(kernelFile, "	condition_2(   1 );\n");
	fprintf(kernelFile, "	condition_2(   2 );\n");
	fprintf(kernelFile, "	condition_2(   4 );\n");
	fprintf(kernelFile, "	condition_2(   8 );\n");
	fprintf(kernelFile, "	condition_2(  16 );\n");
	fprintf(kernelFile, "	condition_2(  32 );\n");
	fprintf(kernelFile, "	condition_2(  64 );\n");
	fprintf(kernelFile, "	condition_2( 128 );\n");
		
	fprintf(kernelFile, "	offset = offset>>1;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	getOffset();\n");
	fprintf(kernelFile, "	unsigned short t = sum[ai];\n");
	fprintf(kernelFile, "	sum[ai] = sum[bi];\n");
	fprintf(kernelFile, "	sum[bi] += t;\n");

	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	return 0;\n");
	fprintf(kernelFile, "}\n");

	fprintf(kernelFile, "__global__ static void DataToQueueB1_32(unsigned int * device_src , unsigned int *device_result, unsigned short *index2048)\n");
	fprintf(kernelFile, "{\n");
	fprintf(kernelFile, "	__shared__ unsigned char src[PROC_BLOCK_SIZE];\n");
	fprintf(kernelFile, "	__shared__ unsigned short res[PROC_BLOCK_SIZE];\n");
	fprintf(kernelFile, "	__shared__ unsigned short sum[THREADS*2+32];\n");
	fprintf(kernelFile, "	unsigned int *ps = (unsigned int *)src;\n");
	fprintf(kernelFile, "	unsigned int *pd = (unsigned int *)res;\n");
	fprintf(kernelFile, "	unsigned int *_device_result =  device_result;\n");

	fprintf(kernelFile, "	int base_offset = __umul24(gridDim.x, blockIdx.y) + blockIdx.x;\n");
	fprintf(kernelFile, "	unsigned int mask_pos = (base_offset<<8) + (threadIdx.y<<4) + threadIdx.x;\n");
	fprintf(kernelFile, "	unsigned int tmp = device_src[mask_pos]; \n");
	fprintf(kernelFile, "	ps[threadIdx.x]  =  ((tmp&0x80808080)>>7);\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	int i2 = threadIdx.x << 1;\n");
	fprintf(kernelFile, "	int i4 = threadIdx.x << 2;\n");
	fprintf(kernelFile, "	int bankOffset = CONFLICT_FREE_OFFSET(i2);\n");
	fprintf(kernelFile, "	sum[i2 + bankOffset] = src[i4] + src[i4 + 1];\n");
	fprintf(kernelFile, "	sum[i2 + bankOffset + 1] = src[i4 + 2] + src[i4 + 3];\n");
	fprintf(kernelFile, "	res[i4] = 2048;\n");
	fprintf(kernelFile, "	res[i4+1] = 2048;\n");
	fprintf(kernelFile, "	res[i4+2] = 2048;\n");
	fprintf(kernelFile, "	res[i4+3] = 2048;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	scan(sum, &index2048[base_offset]);\n");
	fprintf(kernelFile, "	unsigned int bp = sum[i2 + bankOffset];\n");
	fprintf(kernelFile, "	unsigned int c  = src[i4];\n");
	fprintf(kernelFile, "	if (src[i4])\n");
	fprintf(kernelFile, "		res[bp] = i4;\n");
	fprintf(kernelFile, "	if (src[i4 + 1])\n");
	fprintf(kernelFile, "		res[bp + c] = i4 + 1;\n");
	fprintf(kernelFile, "	bp = sum[i2 + bankOffset + 1];\n");
	fprintf(kernelFile, "	c  = src[i4 + 2];\n");
	fprintf(kernelFile, "	if (src[i4 + 2])\n");
	fprintf(kernelFile, "		res[bp] = i4 + 2;\n");
	fprintf(kernelFile, "	if (src[i4 + 3])\n");
	fprintf(kernelFile, "		res[bp + c] = i4 + 3;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	base_offset = ( base_offset << 9 ) + threadIdx.x;\n");
	fprintf(kernelFile, "	_device_result[base_offset] = pd[threadIdx.x];\n");
	fprintf(kernelFile, "	_device_result[base_offset + 256] = pd[threadIdx.x + 256];\n");
	fprintf(kernelFile, "}\n");
	fprintf(kernelFile, "__global__ static void DataToQueueB(unsigned int * mask , unsigned int *old_plan, unsigned int *new_plan, unsigned short *index2048 )\n");
	fprintf(kernelFile, "{\n");
	fprintf(kernelFile, "	__shared__ unsigned char src[PROC_BLOCK_SIZE];\n");
	fprintf(kernelFile, "	__shared__ unsigned short res[PROC_BLOCK_SIZE];\n");
	fprintf(kernelFile, "	__shared__ unsigned short _old_plan[PROC_BLOCK_SIZE];\n");
	fprintf(kernelFile, "	__shared__ unsigned short sum[THREADS*2+32];\n");
	fprintf(kernelFile, "	unsigned int *ps = (unsigned int *)src;\n");
	fprintf(kernelFile, "	int base_offset = __umul24(gridDim.x, blockIdx.y) + blockIdx.x;\n");
	fprintf(kernelFile, "	unsigned int mask_pos = (base_offset<<8) + (threadIdx.y<<4) + threadIdx.x;\n");
	fprintf(kernelFile, "	unsigned int tmp = mask[mask_pos]; \n");
	fprintf(kernelFile, "	ps[threadIdx.x]  =  ((tmp&0x80808080)>>7);\n");
	fprintf(kernelFile, "	unsigned int old_plan_pos = (base_offset<<9) + (threadIdx.y<<4) + threadIdx.x;\n");
	fprintf(kernelFile, "	unsigned int *_old_planI = (unsigned int *)_old_plan;\n");
	fprintf(kernelFile, "	_old_planI[threadIdx.x] = old_plan[old_plan_pos];\n");
	fprintf(kernelFile, "	_old_planI[threadIdx.x+256] = old_plan[old_plan_pos+256];\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	int i2 = threadIdx.x<<1;\n");
	fprintf(kernelFile, "	int i4 = threadIdx.x<<2;\n");
	fprintf(kernelFile, "	int bankOffset = CONFLICT_FREE_OFFSET(i2);\n");
	fprintf(kernelFile, "	sum[i2 + bankOffset] = src[i4] + src[i4 + 1];\n");
	fprintf(kernelFile, "	sum[i2 + bankOffset + 1] = src[i4 + 2] + src[i4 + 3];\n");
	fprintf(kernelFile, "	res[i4] = 2048;\n");
	fprintf(kernelFile, "	res[i4+1] = 2048;\n");
	fprintf(kernelFile, "	res[i4+2] = 2048;\n");
	fprintf(kernelFile, "	res[i4+3] = 2048;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	scan(sum, &index2048[base_offset]);\n");
	fprintf(kernelFile, "	unsigned int bp = sum[i2 + bankOffset];\n");
	fprintf(kernelFile, "	unsigned int c  = src[i4];\n");
	fprintf(kernelFile, "	if (src[i4])\n");
	fprintf(kernelFile, "		res[bp] = i4;\n");
	fprintf(kernelFile, "	if (src[i4 + 1])\n");
	fprintf(kernelFile, "		res[bp + c] = i4 + 1;\n");
	fprintf(kernelFile, "	bp = sum[i2 + bankOffset + 1];\n");
	fprintf(kernelFile, "	c  = src[i4 + 2];\n");
	fprintf(kernelFile, "	if (src[i4 + 2])\n");
	fprintf(kernelFile, "		res[bp] = i4 + 2;\n");
	fprintf(kernelFile, "	if (src[i4 + 3])\n");
	fprintf(kernelFile, "		res[bp + c] = i4 + 3;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	unsigned int device_res_pos = (base_offset<<9) + threadIdx.x;\n");
	fprintf(kernelFile, "	tmp = 0;\n");
	fprintf(kernelFile, "	unsigned int res_pos = threadIdx.x<<1;\n");
	fprintf(kernelFile, "	if (res[res_pos]<=1024)\n");
	fprintf(kernelFile, "		tmp = _old_plan[res[res_pos]];\n");
	fprintf(kernelFile, "	else\n");
	fprintf(kernelFile, "		tmp = 2048;\n");
	fprintf(kernelFile, "	res_pos += 1;   \n");
	fprintf(kernelFile, "	if (res[res_pos]<=1024)\n");
	fprintf(kernelFile, "		tmp += (_old_plan[res[res_pos]]<<16);		 		\n");
	fprintf(kernelFile, "	else\n");
	fprintf(kernelFile, "		tmp += (2048<<16);\n");
	fprintf(kernelFile, "	new_plan[device_res_pos] = tmp;\n");
	fprintf(kernelFile, "	tmp = 0; \n");
	fprintf(kernelFile, "	res_pos = (threadIdx.x<<1) + 512;   \n");
	fprintf(kernelFile, "	if (res[res_pos]<=1024)\n");
	fprintf(kernelFile, "		tmp = _old_plan[res[res_pos]];		 		\n");
	fprintf(kernelFile, "	else\n");
	fprintf(kernelFile, "		tmp = 2048;\n");
	fprintf(kernelFile, "	res_pos += 1;   \n");
	fprintf(kernelFile, "	if (res[res_pos]<=1024)\n");
	fprintf(kernelFile, "		tmp += (_old_plan[res[res_pos]]<<16);		 		\n");
	fprintf(kernelFile, "	else\n");
	fprintf(kernelFile, "		tmp += (2048<<16);\n");
	fprintf(kernelFile, "	new_plan[device_res_pos + 256] = tmp;\n");
	fprintf(kernelFile, "}\n");
	fprintf(kernelFile, "__global__ static void filter(unsigned int *device_src, unsigned int *device_result, unsigned int img_width)\n");
	fprintf(kernelFile, "{\n");
	fprintf(kernelFile, "	__shared__ unsigned char src[32 * 256];\n");
	fprintf(kernelFile, "	unsigned int *ps = (unsigned int *)src;\n");
	fprintf(kernelFile, "	unsigned int PosX = (blockIdx.y << 1) +((threadIdx.x >> 6) & 0x1);\n");
	fprintf(kernelFile, "	unsigned int ps_pos = ((threadIdx.x & 0x80)) + ((threadIdx.x & 0x3C) << 1) + ((threadIdx.x >> 4) & 0x4)  + (threadIdx.x & 0x3);\n");
	fprintf(kernelFile, "	int blockIndex = blockIdx.x << 3;\n");
	fprintf(kernelFile, "	int basePosY = (blockIndex << 1) + (threadIdx.x >> 7);\n");
	fprintf(kernelFile, "	int devSrcOffset = (PosX << 6) + ( threadIdx.x & 0x3F);\n");
	fprintf(kernelFile, "	int mulDevSrc = ( gridDim.y << 1 ) << 6;\n");
	fprintf(kernelFile, "	unsigned int device_src_pos = __umul24(basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(2 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(4 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 2 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(6 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 3 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(8 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 4 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(10 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 5 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(12 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 6 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	device_src_pos = __umul24(14 + basePosY,  mulDevSrc) + devSrcOffset;\n");
	fprintf(kernelFile, "	ps[ps_pos + 7 * 256] = device_src[device_src_pos];\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	int pos = threadIdx.x << 5;\n");
	fprintf(kernelFile, "	if (src[pos     ] > 0) src[pos + 1] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  1] > 0) src[pos + 2] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  2] > 0) src[pos + 3] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  3] > 0) src[pos + 4] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  4] > 0) src[pos + 5] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  5] > 0) src[pos + 6] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  6] > 0) src[pos + 7] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  7] > 0) src[pos + 8] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  8] > 0) src[pos + 9] = 0;\n");
	fprintf(kernelFile, "	if (src[pos +  9] > 0) src[pos +10] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 10] > 0) src[pos +11] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 11] > 0) src[pos +12] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 12] > 0) src[pos +13] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 13] > 0) src[pos +14] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 14] > 0) src[pos +15] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 15] > 0) src[pos +16] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 16] > 0) src[pos +17] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 17] > 0) src[pos +18] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 18] > 0) src[pos +19] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 19] > 0) src[pos +20] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 20] > 0) src[pos +21] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 21] > 0) src[pos +22] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 22] > 0) src[pos +23] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 23] > 0) src[pos +24] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 24] > 0) src[pos +25] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 25] > 0) src[pos +26] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 26] > 0) src[pos +27] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 27] > 0) src[pos +28] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 28] > 0) src[pos +29] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 29] > 0) src[pos +30] = 0;\n");
	fprintf(kernelFile, "	if (src[pos + 30] > 0) src[pos +31] = 0;\n");
	fprintf(kernelFile, "	__syncthreads();\n");
	fprintf(kernelFile, "	int baseOffset = ( ( __umul24(blockIndex, gridDim.y) + blockIdx.y ) << 8 ) + threadIdx.x;\n");
	fprintf(kernelFile, "	int offset = gridDim.y << 8;\n");
	fprintf(kernelFile, "	device_result[baseOffset] = ps[threadIdx.x];\n");
	fprintf(kernelFile, "	device_result[baseOffset + offset]				= ps[threadIdx.x + 256];\n");
	fprintf(kernelFile, "	device_result[baseOffset + ( offset << 1 )]		= ps[threadIdx.x + 256*2];\n");
	fprintf(kernelFile, "	device_result[baseOffset + __umul24(offset, 3)]	= ps[threadIdx.x + 256*3];\n");
	fprintf(kernelFile, "	device_result[baseOffset + ( offset << 2 )]	    = ps[threadIdx.x + 256*4];\n");
	fprintf(kernelFile, "	device_result[baseOffset + __umul24(offset, 5)] = ps[threadIdx.x + 256*5];\n");
	fprintf(kernelFile, "	device_result[baseOffset + __umul24(offset, 6)] = ps[threadIdx.x + 256*6];\n");
	fprintf(kernelFile, "	device_result[baseOffset + __umul24(offset, 7)] = ps[threadIdx.x + 256*7];\n");
	fprintf(kernelFile, "}\n");
}

// управляющая генератором ф-ция
// cascade - сам классификатор
// cascadeStart		- начальный каскад классификатора (начинается с 0)
// cascadeStop		- (финальный каскад классификатора, который требуется учитывать, + 1). Минимум = 1
// cascadePerKernel	- число каскадов, которые будут обрабатываться одним ядром
// threadsPerPixel	- число тредов на один элемент изображения. 1 соответствует haar_first_stage
int cudaGenerator::GenerateCascade(void* _cascade, int cascadeStart, int cascadeStop, int cascadePerKernel, int threadsPerPixel,
					 bool **wasGen, FILE *kernelFile, FILE *JITFile, char ***functionName, int position)
{
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade *) _cascade;
	if (threadsPerPixel == 1)
	{
		// first_cascade -> cascade->stage_classifier[first_cascade]
		// cascadeCount используется как ограничитель в циклах
		
		// ошибка. Первых 2 каскада обрабатывается одним ядром
		if ( cascadePerKernel == 1 )
			return -1;

		// проверяем, было ли ядро уже сгенерировано
		if ( wasGen[1][cascadeStart] == true )
			return -1;

		getCascades(cascade, cascadeStart, cascadeStop - cascadeStart, 16);
		generateInitialization_FirstStage(kernelFile);
		generateMiddle_FirstStage(cascadeStop - cascadeStart, kernelFile);
		generateFinalization_FirstStage(kernelFile);
		fprintf(kernelFile, "\n\n");
		wasGen[1][cascadeStart] = true;
	}
	else
	{
		getCascades(cascade, cascadeStart, cascadeStop - cascadeStart, 32);
		for (int j = 0; j < cascadeStop - cascadeStart; j += cascadePerKernel)
		{
			if (JITFile != NULL)
				fprintf(JITFile, "	&haar_next_stage_%d_%d,", cascadePerKernel, cascadeStart + j);
			if (functionName != NULL)
			{
				functionName[0][position] = new char[30];
				sprintf(functionName[0][position], "haar_next_stage_%d_%d", cascadePerKernel, cascadeStart + j);
				position++;
			}
			if (wasGen[cascadePerKernel - 1][cascadeStart + j] == true)
				continue;

			generateInitialization(cascadeStart + j, threadsPerPixel, cascadePerKernel, kernelFile);
			generateMiddle(j, threadsPerPixel, cascadePerKernel, kernelFile);
			generateFinalization(threadsPerPixel, kernelFile);
			fprintf(kernelFile, "\n\n");

			wasGen[cascadePerKernel - 1][cascadeStart + j] = true;
		}
	}
	free(__gpu_rect2s);
	free(__gpu_rect3s);
	free(__haar_rect2_weights);
	free(__haar_rect3_weights);
	free(__haar_nodes);
	free(__haar_cascade);
	return position;
}
