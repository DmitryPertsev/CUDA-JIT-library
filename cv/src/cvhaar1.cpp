/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* Haar features calculation */

#include <stdio.h>
#include "svldpgm.h"
#include "_cv.h"

//#include <windows.h>

/* these settings affect the quality of detection: change with care */
#define CV_ADJUST_FEATURES 1
#define CV_ADJUST_WEIGHTS  0

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
}
CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
}
CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode* node;
    float* alpha;
}
CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier* classifier;
    int two_rects;
    
    struct CvHidHaarStageClassifier* next;
    struct CvHidHaarStageClassifier* child;
    struct CvHidHaarStageClassifier* parent;
}
CvHidHaarStageClassifier;


struct CvHidHaarClassifierCascade
{
    int  count;
    int  is_stump_based;
    int  has_tilted_features;
    int  is_tree;
    double inv_window_area;
    CvMat sum, sqsum, tilted;
    CvHidHaarStageClassifier* stage_classifier;
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;

    void** ipp_stages;
};


/* IPP functions for object detection */
icvHaarClassifierInitAlloc_32f_t icvHaarClassifierInitAlloc_32f_p = 0;
icvHaarClassifierFree_32f_t icvHaarClassifierFree_32f_p = 0;
icvApplyHaarClassifier_32s32f_C1R_t icvApplyHaarClassifier_32s32f_C1R_p = 0;
icvRectStdDev_32s32f_C1R_t icvRectStdDev_32s32f_C1R_p = 0;

const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;

static CvHaarClassifierCascade*
icvCreateHaarClassifierCascade( int stage_count )
{
    CvHaarClassifierCascade* cascade = 0;
    
    CV_FUNCNAME( "icvCreateHaarClassifierCascade" );

    __BEGIN__;

    int block_size = sizeof(*cascade) + stage_count*sizeof(*cascade->stage_classifier);

    if( stage_count <= 0 )
        CV_ERROR( CV_StsOutOfRange, "Number of stages should be positive" );

    CV_CALL( cascade = (CvHaarClassifierCascade*)cvAlloc( block_size ));
    memset( cascade, 0, block_size );

    cascade->stage_classifier = (CvHaarStageClassifier*)(cascade + 1);
    cascade->flags = CV_HAAR_MAGIC_VAL;
    cascade->count = stage_count;

    __END__;

    return cascade;
}

static void
icvReleaseHidHaarClassifierCascade( CvHidHaarClassifierCascade** _cascade )
{
    if( _cascade && *_cascade )
    {
        CvHidHaarClassifierCascade* cascade = *_cascade;
        if( cascade->ipp_stages && icvHaarClassifierFree_32f_p )
        {
            int i;
            for( i = 0; i < cascade->count; i++ )
            {
                if( cascade->ipp_stages[i] )
                    icvHaarClassifierFree_32f_p( cascade->ipp_stages[i] );
            }
        }
        cvFree( &cascade->ipp_stages );
        cvFree( _cascade );
    }
}

/* create more efficient internal representation of haar classifier cascade */
static CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{
    CvRect* ipp_features = 0;
    float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
    int* ipp_counts = 0;
	// создаем выходной каскад
    CvHidHaarClassifierCascade* out = 0;

    CV_FUNCNAME( "icvCreateHidHaarClassifierCascade" );

    __BEGIN__;

    int i, j, k, l;
    int datasize;
    int total_classifiers = 0;
    int total_nodes = 0;
    char errorstr[100];
    CvHidHaarClassifier* haar_classifier_ptr;
    CvHidHaarTreeNode* haar_node_ptr;
    CvSize orig_window_size;
    int has_tilted_features = 0;
    int max_count = 0;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_ERROR( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( cascade->hid_cascade )
        CV_ERROR( CV_StsError, "hid_cascade has been already created" );

    if( !cascade->stage_classifier )
        CV_ERROR( CV_StsNullPtr, "" );

    if( cascade->count <= 0 )
        CV_ERROR( CV_StsOutOfRange, "Negative number of cascade stages" );

    orig_window_size = cascade->orig_window_size;
    
    /* check input structure correctness and calculate total memory size needed for
       internal representation of the classifier cascade */
	// итого 20 раз. Функция вызывается 1 раз
/*
	Структура каскада			Число эл-тов	Размер в байтах
		cascade						20
			stage_classifier		3
				classifier			2
*/

    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;

        if( !stage_classifier->classifier ||
            stage_classifier->count <= 0 )
        {
            sprintf( errorstr, "header of the stage classifier #%d is invalid "
                     "(has null pointers or non-positive classfier count)", i );
            CV_ERROR( CV_StsError, errorstr );
        }
		// выбрали максимум
        max_count = MAX( max_count, stage_classifier->count );
		// суммируем число классификаторов
        total_classifiers += stage_classifier->count;
		//
        for( j = 0; j < stage_classifier->count; j++ )
        {
			//
            CvHaarClassifier* classifier = stage_classifier->classifier + j;
			// суммируем число узлов
            total_nodes += classifier->count;
            // проходимся по узлам
			for( l = 0; l < classifier->count; l++ )
            {
				// k < 3
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
					//
                    if( classifier->haar_feature[l].rect[k].r.width )
                    {
						// выделяем некоторые прямоугольные области
                        CvRect r = classifier->haar_feature[l].rect[k].r;
                        // распознаем наклоненное лицо
						int tilted = classifier->haar_feature[l].tilted;
                        // имеются данные для наклоненного лица
						has_tilted_features |= tilted != 0;
                        // какой-то размер не задан?
						if( r.width < 0 || r.height < 0 || r.y < 0 ||
							// превышаем размер окна?
                            r.x + r.width > orig_window_size.width
                            ||
							// нарушен размер для рамки для не наклоненного лица?
                            (!tilted &&
                            (r.x < 0 || r.y + r.height > orig_window_size.height))
                            ||
							// нарушен размер для рамки для наклоненного лица?
                            (tilted && (r.x - r.height < 0 ||
                            r.y + r.width + r.height > orig_window_size.height)))
                        {
                            sprintf( errorstr, "rectangle #%d of the classifier #%d of "
                                     "the stage classifier #%d is not inside "
                                     "the reference (original) cascade window", k, j, i );
                            CV_ERROR( CV_StsNullPtr, errorstr );
                        }
                    }
                }
            }
        }
    }

    // для классификатора по-умолчанию получили:
	/*
		Размер данных datasize:
				sizeof(CvHidHaarClassifierCascade)	152
				sizeof(CvHidHaarStageClassifier)	28
				cascade->count						20
				sizeof(CvHidHaarClassifier)			12
				total_classifiers					1047
				sizeof(CvHidHaarTreeNode)			72
				total_nodes							2094
                sizeof(void*)						4

			Итого:									176608 байт
	*/
	datasize = sizeof(CvHidHaarClassifierCascade) +
               sizeof(CvHidHaarStageClassifier)*cascade->count +
               sizeof(CvHidHaarClassifier) * total_classifiers +
               sizeof(CvHidHaarTreeNode) * total_nodes +
               sizeof(void*)*(total_nodes + total_classifiers);

	// выделяем память и обнуляем
    CV_CALL( out = (CvHidHaarClassifierCascade*)cvAlloc( datasize ));
    memset( out, 0, sizeof(*out) );

    /* init header */
	// сораняем число каскадов
    out->count = cascade->count;
	// 
    out->stage_classifier = (CvHidHaarStageClassifier*)(out + 1);
	//
    haar_classifier_ptr = (CvHidHaarClassifier*)(out->stage_classifier + cascade->count);
	//
    haar_node_ptr = (CvHidHaarTreeNode*)(haar_classifier_ptr + total_classifiers);

	//
    out->is_stump_based = 1;
	//
    out->has_tilted_features = has_tilted_features;
	//
    out->is_tree = 0;

    // инициализируем внутреннее представление
    for( i = 0; i < cascade->count; i++ )
    {
		// выделяем очередной элемент исх. каскада и результирующего
		// sizeof(CvHaarStageClassifier)		= 24
		// sizeof(CvHidHaarStageClassifier)		= 28
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
        CvHidHaarStageClassifier* hid_stage_classifier = out->stage_classifier + i;

		// сохр. число элементов в текущем слое
        hid_stage_classifier->count = stage_classifier->count;
		// const float icv_stage_threshold_bias = 0.0001f;
		// сохраняем порог
        hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
        hid_stage_classifier->classifier = haar_classifier_ptr;
        hid_stage_classifier->two_rects = 1;
		// смещаемся на следующий блок классификаторов
        haar_classifier_ptr += stage_classifier->count;

		// сохраняем информацию из анализируемого каскада о parent
        hid_stage_classifier->parent = (stage_classifier->parent == -1)
            ? NULL : out->stage_classifier + stage_classifier->parent;
		//			о следующем узле
        hid_stage_classifier->next = (stage_classifier->next == -1)
            ? NULL : out->stage_classifier + stage_classifier->next;
		//			о child
        hid_stage_classifier->child = (stage_classifier->child == -1)
            ? NULL : out->stage_classifier + stage_classifier->child;
        
		// дерево?
        out->is_tree |= hid_stage_classifier->next != NULL;

		// обрабатывает классификаторы каскада i
        for( j = 0; j < stage_classifier->count; j++ )
        {
			// извлекаем очередной элемент классификатора
            CvHaarClassifier* classifier = stage_classifier->classifier + j;
			//   и место сохранения в результирующем массиве
            CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
			// сохр. число узлов заданного классификатора
            int node_count = classifier->count;
			//
            float* alpha_ptr = (float*)(haar_node_ptr + node_count);

			//
            hid_classifier->count = node_count;
            hid_classifier->node = haar_node_ptr;
            hid_classifier->alpha = alpha_ptr;
            
			//
            for( l = 0; l < node_count; l++ )
            {
				// извлекаем очередной узел дерева
                CvHidHaarTreeNode* node = hid_classifier->node + l;
				// 
                CvHaarFeature* feature = classifier->haar_feature + l;
                memset( node, -1, sizeof(*node) );
				//
                node->threshold = classifier->threshold[l];
                node->left = classifier->left[l];
                node->right = classifier->right[l];

				//#define DBL_EPSILON     2.2204460492503131e-016 /* smallest such that 1.0+DBL_EPSILON != 1.0 */
                if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
                    feature->rect[2].r.width == 0 ||
                    feature->rect[2].r.height == 0 )
                    memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
                else
                    hid_stage_classifier->two_rects = 0;
            }

			//
            memcpy( alpha_ptr, classifier->alpha, (node_count+1)*sizeof(alpha_ptr[0]));
			//
            haar_node_ptr =
                (CvHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));

			//
            out->is_stump_based &= node_count == 1;
        }
    }

    //
    // NOTE: Currently, OpenMP is implemented and IPP modes are incompatible.
    // 
#ifndef _OPENMP
    {
		//
    int can_use_ipp = icvHaarClassifierInitAlloc_32f_p != 0 &&
        icvHaarClassifierFree_32f_p != 0 &&
                      icvApplyHaarClassifier_32s32f_C1R_p != 0 &&
                      icvRectStdDev_32s32f_C1R_p != 0 &&
                      !out->has_tilted_features && !out->is_tree && out->is_stump_based;

	// никогда не выполняется, т.к. icvHaarClassifierInitAlloc_32f_p = 0
    if( can_use_ipp )
    {	// не используется
        int ipp_datasize = cascade->count*sizeof(out->ipp_stages[0]);
        float ipp_weight_scale=(float)(1./((orig_window_size.width-icv_object_win_border*2)*
            (orig_window_size.height-icv_object_win_border*2)));

        CV_CALL( out->ipp_stages = (void**)cvAlloc( ipp_datasize ));
        memset( out->ipp_stages, 0, ipp_datasize );

        CV_CALL( ipp_features = (CvRect*)cvAlloc( max_count*3*sizeof(ipp_features[0]) ));
        CV_CALL( ipp_weights = (float*)cvAlloc( max_count*3*sizeof(ipp_weights[0]) ));
        CV_CALL( ipp_thresholds = (float*)cvAlloc( max_count*sizeof(ipp_thresholds[0]) ));
        CV_CALL( ipp_val1 = (float*)cvAlloc( max_count*sizeof(ipp_val1[0]) ));
        CV_CALL( ipp_val2 = (float*)cvAlloc( max_count*sizeof(ipp_val2[0]) ));
        CV_CALL( ipp_counts = (int*)cvAlloc( max_count*sizeof(ipp_counts[0]) ));

        for( i = 0; i < cascade->count; i++ )
        {
            CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
            for( j = 0, k = 0; j < stage_classifier->count; j++ )
            {
                CvHaarClassifier* classifier = stage_classifier->classifier + j;
                int rect_count = 2 + (classifier->haar_feature->rect[2].r.width != 0);

                ipp_thresholds[j] = classifier->threshold[0];
                ipp_val1[j] = classifier->alpha[0];
                ipp_val2[j] = classifier->alpha[1];
                ipp_counts[j] = rect_count;
                
                for( l = 0; l < rect_count; l++, k++ )
                {
                    ipp_features[k] = classifier->haar_feature->rect[l].r;
                    //ipp_features[k].y = orig_window_size.height - ipp_features[k].y - ipp_features[k].height;
                    ipp_weights[k] = classifier->haar_feature->rect[l].weight*ipp_weight_scale;
                }
            }
            
            if( icvHaarClassifierInitAlloc_32f_p( &out->ipp_stages[i],
                ipp_features, ipp_weights, ipp_thresholds,
                ipp_val1, ipp_val2, ipp_counts, stage_classifier->count ) < 0 )
                break;
        }

        if( i < cascade->count )
        {
            for( j = 0; j < i; j++ )
                if( icvHaarClassifierFree_32f_p && out->ipp_stages[i] )
                    icvHaarClassifierFree_32f_p( out->ipp_stages[i] );
            cvFree( &out->ipp_stages );
        }
    }
    }
#endif

    cascade->hid_cascade = out;
    assert( (char*)haar_node_ptr - (char*)out <= datasize );

    __END__;

    if( cvGetErrStatus() < 0 )
        icvReleaseHidHaarClassifierCascade( &out );

    cvFree( &ipp_features );
    cvFree( &ipp_weights );
    cvFree( &ipp_thresholds );
    cvFree( &ipp_val1 );
    cvFree( &ipp_val2 );
    cvFree( &ipp_counts );

    return out;
}


#define sum_elem_ptr(sum,row,col)  \
    ((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
    ((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))

#define calc_sum(rect,offset) \
    ((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])


CV_IMPL void
cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* _cascade,
                                     const CvArr* _sum,
                                     const CvArr* _sqsum,
                                     const CvArr* _tilted_sum,
                                     double scale )
{
    CV_FUNCNAME("cvSetImagesForHaarClassifierCascade");

    __BEGIN__;

    CvMat sum_stub, *sum = (CvMat*)_sum;
    CvMat sqsum_stub, *sqsum = (CvMat*)_sqsum;
    CvMat tilted_stub, *tilted = (CvMat*)_tilted_sum;
    CvHidHaarClassifierCascade* cascade;
    int coi0 = 0, coi1 = 0;
    int i;
    CvRect equ_rect;
    double weight_scale;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_ERROR( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( scale <= 0 )
        CV_ERROR( CV_StsOutOfRange, "Scale must be positive" );
// Проебразуем CvArr (IplImage или CvMat,...) в CvMat.
    CV_CALL( sum = cvGetMat( sum, &sum_stub, &coi0 ));
    CV_CALL( sqsum = cvGetMat( sqsum, &sqsum_stub, &coi1 ));

    if( coi0 || coi1 )
        CV_ERROR( CV_BadCOI, "COI is not supported" );

    if( !CV_ARE_SIZES_EQ( sum, sqsum ))
        CV_ERROR( CV_StsUnmatchedSizes, "All integral images must have the same size" );

    if( CV_MAT_TYPE(sqsum->type) != CV_64FC1 ||
        CV_MAT_TYPE(sum->type) != CV_32SC1 )
        CV_ERROR( CV_StsUnsupportedFormat,
        "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );
	// внутреннее представление было раньше создано?
    if( !_cascade->hid_cascade )
        CV_CALL( icvCreateHidHaarClassifierCascade(_cascade) );

    cascade = _cascade->hid_cascade;

    if( cascade->has_tilted_features )
    {	// не выполняется
        CV_CALL( tilted = cvGetMat( tilted, &tilted_stub, &coi1 ));

        if( CV_MAT_TYPE(tilted->type) != CV_32SC1 )
            CV_ERROR( CV_StsUnsupportedFormat,
            "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

        if( sum->step != tilted->step )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Sum and tilted_sum must have the same stride (step, widthStep)" );

        if( !CV_ARE_SIZES_EQ( sum, tilted ))
            CV_ERROR( CV_StsUnmatchedSizes, "All integral images must have the same size" );
        cascade->tilted = *tilted;
    }
    // сохр. новые значения в исх. матрицу
    _cascade->scale = scale;
    _cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );
    _cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );
	// сохр. в новый каскад суммы, переданные извне
    cascade->sum = *sum;
    cascade->sqsum = *sqsum;
    // рассчитываем значения новой рамки
    equ_rect.x = equ_rect.y = cvRound(scale);
    equ_rect.width = cvRound((_cascade->orig_window_size.width-2)*scale);
    equ_rect.height = cvRound((_cascade->orig_window_size.height-2)*scale);
	// рассчитали масштабирующий коэффициент
    weight_scale = 1./(equ_rect.width*equ_rect.height);
    cascade->inv_window_area = weight_scale;
/*
#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
    (assert( (unsigned)(row) < (unsigned)(mat).rows &&   \
             (unsigned)(col) < (unsigned)(mat).cols ),   \
     (mat).data.ptr + (size_t)(mat).step*(row) + (pix_size)*(col))

#define sum_elem_ptr(sum,row,col)  \
    ((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
    ((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))
*/
    cascade->p0 = sum_elem_ptr(*sum, equ_rect.y, equ_rect.x);
    cascade->p1 = sum_elem_ptr(*sum, equ_rect.y, equ_rect.x + equ_rect.width );
    cascade->p2 = sum_elem_ptr(*sum, equ_rect.y + equ_rect.height, equ_rect.x );
    cascade->p3 = sum_elem_ptr(*sum, equ_rect.y + equ_rect.height,
                                     equ_rect.x + equ_rect.width );

    cascade->pq0 = sqsum_elem_ptr(*sqsum, equ_rect.y, equ_rect.x);
    cascade->pq1 = sqsum_elem_ptr(*sqsum, equ_rect.y, equ_rect.x + equ_rect.width );
    cascade->pq2 = sqsum_elem_ptr(*sqsum, equ_rect.y + equ_rect.height, equ_rect.x );
    cascade->pq3 = sqsum_elem_ptr(*sqsum, equ_rect.y + equ_rect.height,
                                          equ_rect.x + equ_rect.width );

    /* init pointers in haar features according to real window size and
       given image pointers */
    {
#ifdef _OPENMP
    int max_threads = cvGetNumThreads();
    #pragma omp parallel for num_threads(max_threads), schedule(dynamic) 
#endif // _OPENMP
    for( i = 0; i < _cascade->count; i++ )
    {
        int j, k, l;
        for( j = 0; j < cascade->stage_classifier[i].count; j++ )
        {
            for( l = 0; l < cascade->stage_classifier[i].classifier[j].count; l++ )
            {
				// организуем перебор всего массива вплоть до классификатора каждого элемента каскада
				// извлекаем "характерные черты"
                CvHaarFeature* feature = 
                    &_cascade->stage_classifier[i].classifier[j].haar_feature[l];
				// извлекаем "характерные черты" скрытого слоя
                CvHidHaarFeature* hidfeature = 
                    &cascade->stage_classifier[i].classifier[j].node[l].feature;
                double sum0 = 0, area0 = 0;
                CvRect r[3];
#if CV_ADJUST_FEATURES
				// иниц. переменные
                int base_w = -1, base_h = -1;
                int new_base_w = 0, new_base_h = 0;
                int kx, ky;
                int flagx = 0, flagy = 0;
                int x0 = 0, y0 = 0;
#endif
                int nr;

                /* align blocks */ // k < 3
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if( !hidfeature->rect[k].p0 )
                        break;
#if CV_ADJUST_FEATURES
					// берем очередную область
                    r[k] = feature->rect[k].r;
					// сохраняем минимум
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].width-1) );
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].x - r[0].x-1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].height-1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].y - r[0].y-1) );
#endif
                }

                nr = k;

#if CV_ADJUST_FEATURES
				// корректируем
                base_w += 1;
                base_h += 1;
                kx = r[0].width / base_w;
                ky = r[0].height / base_h;

				//
                if( kx <= 0 )
                {	// не используется
                    flagx = 1;
                    new_base_w = cvRound( r[0].width * scale ) / kx;
                    x0 = cvRound( r[0].x * scale );
                }

				//
                if( ky <= 0 )
                {	// не используется
                    flagy = 1;
                    new_base_h = cvRound( r[0].height * scale ) / ky;
                    y0 = cvRound( r[0].y * scale );
                }
#endif
        
				//
                for( k = 0; k < nr; k++ )
                {
                    CvRect tr;
                    double correction_ratio;
            
#if CV_ADJUST_FEATURES
					//
                    if( flagx )
                    {	// не используется
                        tr.x = (r[k].x - r[0].x) * new_base_w / base_w + x0;
                        tr.width = r[k].width * new_base_w / base_w;
                    }
                    else
#endif
                    {
						// модифицируем прямоугольную область по оси X
                        tr.x = cvRound( r[k].x * scale );
                        tr.width = cvRound( r[k].width * scale );
                    }

#if CV_ADJUST_FEATURES
					//
                    if( flagy )
                    {	// не используется
                        tr.y = (r[k].y - r[0].y) * new_base_h / base_h + y0;
                        tr.height = r[k].height * new_base_h / base_h;
                    }
                    else
#endif
                    {
						// модифицируем прямоугольную область по оси Y
                        tr.y = cvRound( r[k].y * scale );
                        tr.height = cvRound( r[k].height * scale );
                    }

#if CV_ADJUST_WEIGHTS
                    {// не исполняется
                    // RAINER START
                    const float orig_feature_size =  (float)(feature->rect[k].r.width)*feature->rect[k].r.height; 
                    const float orig_norm_size = (float)(_cascade->orig_window_size.width)*(_cascade->orig_window_size.height);
                    const float feature_size = float(tr.width*tr.height);
                    //const float normSize    = float(equ_rect.width*equ_rect.height);
                    float target_ratio = orig_feature_size / orig_norm_size;
                    //float isRatio = featureSize / normSize;
                    //correctionRatio = targetRatio / isRatio / normSize;
                    correction_ratio = target_ratio / feature_size;
                    // RAINER END
                    }
#else
					// корректирующий коэффициент
                    correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
#endif

					// если каскад не для наклонных лиц
                    if( !feature->tilted )
                    {
						// модифицируем область
                        hidfeature->rect[k].p0 = sum_elem_ptr(*sum, tr.y, tr.x);
                        hidfeature->rect[k].p1 = sum_elem_ptr(*sum, tr.y, tr.x + tr.width);
                        hidfeature->rect[k].p2 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x);
                        hidfeature->rect[k].p3 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x + tr.width);
                    }
                    else
                    {	// не используется
						//
                        hidfeature->rect[k].p2 = sum_elem_ptr(*tilted, tr.y + tr.width, tr.x + tr.width);
                        hidfeature->rect[k].p3 = sum_elem_ptr(*tilted, tr.y + tr.width + tr.height,
                                                              tr.x + tr.width - tr.height);
                        hidfeature->rect[k].p0 = sum_elem_ptr(*tilted, tr.y, tr.x);
                        hidfeature->rect[k].p1 = sum_elem_ptr(*tilted, tr.y + tr.height, tr.x - tr.height);
                    }

					// ширина области
                    hidfeature->rect[k].weight = (float)(feature->rect[k].weight * correction_ratio);

					//
                    if( k == 0 )
                        area0 = tr.width * tr.height;
                    else
                        sum0 += hidfeature->rect[k].weight * tr.width * tr.height;
                }

				//
                hidfeature->rect[0].weight = (float)(-sum0/area0);
            } /* l */
        } /* j */
    }
    }

    __END__;
}


CV_INLINE
double icvEvalHidHaarClassifier( CvHidHaarClassifier* classifier,
                                 double variance_norm_factor,
                                 size_t p_offset )
{
    int idx = 0;
    do 
    {
        CvHidHaarTreeNode* node = classifier->node + idx;
        double t = node->threshold * variance_norm_factor;

        double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
        sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

        if( node->feature.rect[2].p0 )
            sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

        idx = sum < t ? node->left : node->right;
    }
    while( idx > 0 );
    return classifier->alpha[-idx];
}


CV_IMPL int
cvRunHaarClassifierCascade( CvHaarClassifierCascade* _cascade,
                            CvPoint pt, int start_stage )
{
    int result = -1;
    CV_FUNCNAME("cvRunHaarClassifierCascade");

    __BEGIN__;

    int p_offset, pq_offset;
    int i, j;
    double mean, variance_norm_factor;
    CvHidHaarClassifierCascade* cascade;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_ERROR( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid cascade pointer" );

    cascade = _cascade->hid_cascade;
    if( !cascade )
        CV_ERROR( CV_StsNullPtr, "Hidden cascade has not been created.\n"
            "Use cvSetImagesForHaarClassifierCascade" );

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + _cascade->real_window_size.width >= cascade->sum.width-2 ||
        pt.y + _cascade->real_window_size.height >= cascade->sum.height-2 )
        EXIT;

    p_offset = pt.y * (cascade->sum.step/sizeof(sumtype)) + pt.x;
    pq_offset = pt.y * (cascade->sqsum.step/sizeof(sqsumtype)) + pt.x;
    mean = calc_sum(*cascade,p_offset)*cascade->inv_window_area;
    variance_norm_factor = cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
                           cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
    variance_norm_factor = variance_norm_factor*cascade->inv_window_area - mean*mean;
    if( variance_norm_factor >= 0. )
        variance_norm_factor = sqrt(variance_norm_factor);
    else
        variance_norm_factor = 1.;

    if( cascade->is_tree )
    {
        CvHidHaarStageClassifier* ptr;
        assert( start_stage == 0 );

        result = 1;
        ptr = cascade->stage_classifier;

        while( ptr )
        {
            double stage_sum = 0;

            for( j = 0; j < ptr->count; j++ )
            {
                stage_sum += icvEvalHidHaarClassifier( ptr->classifier + j,
                    variance_norm_factor, p_offset );
            }

            if( stage_sum >= ptr->threshold )
            {
                ptr = ptr->child;
            }
            else
            {
                while( ptr && ptr->next == NULL ) ptr = ptr->parent;
                if( ptr == NULL )
                {
                    result = 0;
                    EXIT;
                }
                ptr = ptr->next;
            }
        }
    }
    else if( cascade->is_stump_based )
    {
        for( i = start_stage; i < cascade->count; i++ )
        {
            double stage_sum = 0;

            if( cascade->stage_classifier[i].two_rects )
            {
                for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                {
                    CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                    CvHidHaarTreeNode* node = classifier->node;
                    double sum, t = node->threshold*variance_norm_factor, a, b;

                    sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                    sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

                    a = classifier->alpha[0];
                    b = classifier->alpha[1];
                    stage_sum += sum < t ? a : b;
                }
            }
            else
            {
                for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                {
                    CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                    CvHidHaarTreeNode* node = classifier->node;
                    double sum, t = node->threshold*variance_norm_factor, a, b;

                    sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                    sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

                    if( node->feature.rect[2].p0 )
                        sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

                    a = classifier->alpha[0];
                    b = classifier->alpha[1];
                    stage_sum += sum < t ? a : b;
                }
            }

            if( stage_sum < cascade->stage_classifier[i].threshold )
            {
                result = -i;
                EXIT;
            }
        }
    }
    else
    {
        for( i = start_stage; i < cascade->count; i++ )
        {
            double stage_sum = 0;

            for( j = 0; j < cascade->stage_classifier[i].count; j++ )
            {
                stage_sum += icvEvalHidHaarClassifier(
                    cascade->stage_classifier[i].classifier + j,
                    variance_norm_factor, p_offset );
            }

            if( stage_sum < cascade->stage_classifier[i].threshold )
            {
                result = -i;
                EXIT;
            }
        }
    }

    result = 1;

    __END__;

    return result;
}


static int is_equal( const void* _r1, const void* _r2, void* )
{
    const CvRect* r1 = (const CvRect*)_r1;
    const CvRect* r2 = (const CvRect*)_r2;
    int distance = cvRound(r1->width*0.2);

    return r2->x <= r1->x + distance &&
           r2->x >= r1->x - distance &&
           r2->y <= r1->y + distance &&
           r2->y >= r1->y - distance &&
           r2->width <= cvRound( r1->width * 1.2 ) &&
           cvRound( r2->width * 1.2 ) >= r1->width;
}

/*
Detects objects in the image

image			Image to detect objects in. 
cascade			Haar classifier cascade in internal representation. 
storage			Memory storage to store the resultant sequence of the object candidate rectangles. 
scale_factor	The factor by which the search window is scaled between the subsequent scans, for example, 1.1 means increasing window by 10%. 
min_neighbors	Minimum number (minus 1) of neighbor rectangles that makes up an object. All the groups of a smaller number of rectangles than min_neighbors-1 are rejected. If min_neighbors is 0, the function does not any grouping at all and returns all the detected candidate rectangles, which may be useful if the user wants to apply a customized grouping procedure. 
flags			Mode of operation. Currently the only flag that may be specified is CV_HAAR_DO_CANNY_PRUNING. If it is set, the function uses Canny edge detector to reject some image regions that contain too few or too much edges and thus can not contain the searched object. The particular threshold values are tuned for face detection and in this case the pruning speeds up the processing. 
min_size		Minimum window size. By default, it is set to the size of samples the classifier has been trained on (~20×20 for face detection).

The function cvHaarDetectObjects finds rectangular regions in the given image that are likely to contain objects the cascade has been trained for and returns those regions as a sequence of rectangles. The function scans the image several times at different scales (see cvSetImagesForHaarClassifierCascade). Each time it considers overlapping regions in the image and applies the classifiers to the regions using cvRunHaarClassifierCascade. It may also apply some heuristics to reduce number of analyzed regions, such as Canny prunning. After it has proceeded and collected the candidate rectangles (regions that passed the classifier cascade), it groups them and returns a sequence of average rectangles for each large enough group. The default parameters (scale_factor=1.1, min_neighbors=3, flags=0) are tuned for accurate yet slow object detection. For a faster operation on real video images the settings are: scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, min_size=<minimum possible face size> (for example, ~1/4 to 1/16 of the image area in case of video conferencing).
*/
/*
#include <cuda_runtime_api.h>

void _GetSystemInfo__()
{
	int deviceCount;
	cudaDeviceProp deviceProp;

	//Сколько устройств CUDA установлено на PC.
	cudaGetDeviceCount(&deviceCount);

	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++)
	{
		//Получаем информацию об устройстве
		cudaGetDeviceProperties(&deviceProp, i);

		//Выводим иформацию об устройстве
		printf("Device name: %s\n", deviceProp.name);
		printf("Total global memory: %d MB\n", deviceProp.totalGlobalMem / (1024 * 1024) );
		printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Memory pitch: %d\n", deviceProp.memPitch);
		printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

		printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("Max grid size: x = %d, y = %d, z = %d\n", 
			deviceProp.maxGridSize[0], 
			deviceProp.maxGridSize[1], 
			deviceProp.maxGridSize[2]); 

		printf("Clock rate: %d\n", deviceProp.clockRate);
		printf("Total constant memory: %d\n", deviceProp.totalConstMem); 
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Texture alignment: %d\n", deviceProp.textureAlignment);
		printf("Device overlap: %d\n", deviceProp.deviceOverlap);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

		printf("Kernel execution timeout enabled: %s\n\n\n",
			deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
	}
}
*/
CV_IMPL CvSeq*
cvHaarDetectObjects( const CvArr* _img,
                     CvHaarClassifierCascade* cascade,
                     CvMemStorage* storage, double scale_factor,
                     int min_neighbors, int flags, CvSize min_size/*, int posTime, int *iterCounter, DWORD *timeArray */)
{
    int split_stage = 2;
	int posTime;
	int *iterCounter;
	DWORD *timeArray;
    CvMat stub, *img = (CvMat*)_img;
    CvMat *temp = 0, *sum = 0, *tilted = 0, *sqsum = 0, *norm_img = 0, *sumcanny = 0, *img_small = 0;
	CvMat *_temp_1 = 0, *_temp_2 = 0, *_temp_3 = 0;
    CvSeq* seq = 0;
    CvSeq* seq2 = 0;
    CvSeq* idx_seq = 0;
    CvSeq* result_seq = 0;
    CvMemStorage* temp_storage = 0;
    CvAvgComp* comps = 0;
    int i;

	//_GetSystemInfo__();
    
#ifdef _OPENMP
    CvSeq* seq_thread[CV_MAX_THREADS] = {0};
    int max_threads = 0;
#endif
    
    CV_FUNCNAME( "cvHaarDetectObjects" );

    __BEGIN__;

    double factor;
    int npass = 2, coi;
    int do_canny_pruning = flags & CV_HAAR_DO_CANNY_PRUNING;

	// осуществляем проверку каскада cascade
    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_ERROR( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );
			// блок отвечает за осуществление всех проверок
	// память не была выделена?
    if( !storage )
        CV_ERROR( CV_StsNullPtr, "Null storage pointer" );
	// преобразуем CvArr -> CvMat
    CV_CALL( img = cvGetMat( img, &stub, &coi ));
	// ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
	// Для cvIntegral
	//int cn = CV_MAT_CN(img->type);
	// cn = 1
    if( coi )
        CV_ERROR( CV_BadCOI, "COI is not supported" );

    if( CV_MAT_DEPTH(img->type) != CV_8U )
        CV_ERROR( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );
			//
	// выделяет память и инициализирует заголовок CvMat, а также выделяет требуемый объем памяти для размещения данных
	/*
			CV_8UC1		- 8-битное unsigned integer одноканальная матрица
			CV_32SC1	- 32-битное signed integer одноканальная матрица
			CV_64FC1	- 64-битное float одноканальная матрица

		sum и sqsum используется в функции cvIntegral (опис. см. ниже)
		sum		- используется для хранения результатов суммирования
		sqsum	- используется для хранения результатов квадратов (squared) сумм
	*/
    CV_CALL( temp = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	CV_CALL( _temp_1 = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	CV_CALL( _temp_2 = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	CV_CALL( _temp_3 = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
    CV_CALL( sum = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));
    CV_CALL( sqsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_64FC1 ));
	// выделяет память для child у родителя storage
    CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));

#ifdef _OPENMP
    max_threads = cvGetNumThreads();
    for( i = 0; i < max_threads; i++ )
    {
        CvMemStorage* temp_storage_thread;
        CV_CALL( temp_storage_thread = cvCreateMemStorage(0));
        CV_CALL( seq_thread[i] = cvCreateSeq( 0, sizeof(CvSeq),
                        sizeof(CvRect), temp_storage_thread ));
    }
#endif
	double startTime;
    if( !cascade->hid_cascade )
	{
/*
   helper functions for RNG initialization and accurate time measurement:
   uses internal clock counter on x86
*/
		startTime = (double)cvGetTickCount();
		CV_CALL( icvCreateHidHaarClassifierCascade(cascade) );
		startTime = (double)cvGetTickCount() - startTime;
		printf("icvCreateHidHaarClassifierCascade = \t%g ms\n", startTime /( (double)cvGetTickFrequency() * 1000. ));
	}

    if( cascade->hid_cascade->has_tilted_features )
        tilted = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );
	// создает новую пустую последовательность, которая будет находиться в определенном месте (?)
	//		CvRect - структура, опис. начало координат, длину и ширину прямоугольной области
	//		CvAvgComp - структура, включающая CvRect и число ближайшего окружения (?)
    seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), temp_storage );
    seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), temp_storage );
    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

    if( min_neighbors == 0 )
        seq = result_seq;
	// проверяет правильность цветовой кодировки (?)
    if( CV_MAT_CN(img->type) > 1 )
    {
        cvCvtColor( img, temp, CV_BGR2GRAY );
        img = temp;
    }
    // если матрица требует масштабирования(?)
	if( flags & CV_HAAR_SCALE_IMAGE )
	{	// не используется
		CvSize win_size0 = cascade->orig_window_size;
		int use_ipp = cascade->hid_cascade->ipp_stages != 0 &&
			icvApplyHaarClassifier_32s32f_C1R_p != 0;

		if( use_ipp )
			CV_CALL( norm_img = cvCreateMat( img->rows, img->cols, CV_32FC1 ));
		CV_CALL( img_small = cvCreateMat( img->rows + 1, img->cols + 1, CV_8UC1 ));

		for( factor = 1; ; factor *= scale_factor )
		{
			int positive = 0;
			printf("factor = %f\n", factor);
			int x, y;
			CvSize win_size = { cvRound(win_size0.width*factor),
				cvRound(win_size0.height*factor) };
			CvSize sz = { cvRound( img->cols/factor ), cvRound( img->rows/factor ) };
			CvSize sz1 = { sz.width - win_size0.width, sz.height - win_size0.height };
			CvRect rect1 = { icv_object_win_border, icv_object_win_border,
				win_size0.width - icv_object_win_border*2,
				win_size0.height - icv_object_win_border*2 };
			CvMat img1, sum1, sqsum1, norm1, tilted1, mask1;
			CvMat* _tilted = 0;

			if( sz1.width <= 0 || sz1.height <= 0 )
				break;
			if( win_size.width < min_size.width || win_size.height < min_size.height )
				continue;

			img1 = cvMat( sz.height, sz.width, CV_8UC1, img_small->data.ptr );
			sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
			sqsum1 = cvMat( sz.height+1, sz.width+1, CV_64FC1, sqsum->data.ptr );
			if( tilted )
			{
				tilted1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tilted->data.ptr );
				_tilted = &tilted1;
			}
			norm1 = cvMat( sz1.height, sz1.width, CV_32FC1, norm_img ? norm_img->data.ptr : 0 );
			mask1 = cvMat( sz1.height, sz1.width, CV_8UC1, temp->data.ptr );

			// преобразует изображение img в img1. Тип интерполяции: CV_INTER_LINEAR
			double startTimeFactor = (double)cvGetTickCount();
			cvResize( img, &img1, CV_INTER_LINEAR );
			startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
			printf("\tcvResize = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));

			// см. файл cvsumpixels.cpp
			startTimeFactor = (double)cvGetTickCount();
			cvIntegral( &img1, &sum1, &sqsum1, _tilted );
			startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
			printf("\tcvIntegral = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));

			if( use_ipp && icvRectStdDev_32s32f_C1R_p( sum1.data.i, sum1.step,
				sqsum1.data.db, sqsum1.step, norm1.data.fl, norm1.step, sz1, rect1 ) < 0 )
				use_ipp = 0;

			if( use_ipp )
			{		// не используется
				positive = mask1.cols*mask1.rows;
				cvSet( &mask1, cvScalarAll(255) );
				for( i = 0; i < cascade->count; i++ )
				{
					if( icvApplyHaarClassifier_32s32f_C1R_p(sum1.data.i, sum1.step,
						norm1.data.fl, norm1.step, mask1.data.ptr, mask1.step,
						sz1, &positive, cascade->hid_cascade->stage_classifier[i].threshold,
						cascade->hid_cascade->ipp_stages[i]) < 0 )
					{
						use_ipp = 0;
						break;
					}
					if( positive <= 0 )
						break;
				}
			}

			printf("\timg1. width = %d. height = %d\n", img1.width, img1.height);
			printf("\tsz1. width = %d. height = %d\n", sz1.width, sz1.height);
			if( !use_ipp )
			{
				startTimeFactor = (double)cvGetTickCount();
				cvSetImagesForHaarClassifierCascade( cascade, &sum1, &sqsum1, 0, 1. );
				startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
				printf("\tcvSetImagesForHaarClassifierCascade = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));

				startTimeFactor = (double)cvGetTickCount();
				for( y = 0, positive = 0; y < sz1.height; y++ )
					for( x = 0; x < sz1.width; x++ )
					{
						mask1.data.ptr[mask1.step*y + x] =
							cvRunHaarClassifierCascade( cascade, cvPoint(x,y), 0 ) > 0;
						positive += mask1.data.ptr[mask1.step*y + x];
					}
					startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
					printf("\tcycle for. Generate mask1 and positive = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));
			}

			startTimeFactor = (double)cvGetTickCount();
			if( positive > 0 )
			{
				for( y = 0; y < sz1.height; y++ )
					for( x = 0; x < sz1.width; x++ )
						if( mask1.data.ptr[mask1.step*y + x] != 0 )
						{
							CvRect obj_rect = { cvRound(y*factor), cvRound(x*factor),
								win_size.width, win_size.height };
							cvSeqPush( seq, &obj_rect );
						}
			}
			startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
			printf("\tcycle for. Check positive = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));
		}
	}
    else
    {	//Finds integral image: SUM(X,Y) = sum(x<X,y<Y)I(x,y)
		/*
		void cvIntegral( const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL );
image		-	The source image, W×H, 8-bit or floating-point (32f or 64f) image. 
sum			-	The integral image, W+1×H+1, 32-bit integer or double precision floating-point (64f). 
sqsum		-	The integral image for squared pixel values, W+1×H+1, double precision floating-point (64f). 
tilted_sum	-	The integral for the image rotated by 45 degrees, W+1×H+1, the same data type as sum.

sum(X,Y)=sum (x<X,y<Y) image(x,y)
sqsum(X,Y)=sum (x<X,y<Y) image(x,y)^2		canny
tilted_sum(X,Y)=sum (y<Y,abs(x-X)<y) image(x,y)
		*/
		startTime = (double)cvGetTickCount();
        cvIntegral( img, sum, sqsum, tilted );
		startTime = (double)cvGetTickCount() - startTime;
		printf("cvIntegral = \t%g ms\n", startTime /( (double)cvGetTickFrequency() * 1000. ));
		// произвести отсечение ветвей дерева (?)
        if( do_canny_pruning )
        {	// не выполняется
            sumcanny = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );
			/*
Implements Canny algorithm for edge detection

void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size=3 );

image			Input image. 
edges			Image to store the edges found by the function. 
threshold1		первый порого. 
threshold2		второй порог. 
aperture_size	Aperture parameter for Sobel operator (see cvSobel). 

The function cvCanny finds the edges (край) on the input image image and marks them in the output image edges using the Canny algorithm. The smallest of threshold1 and threshold2 is used for edge linking, the largest - to find initial segments of strong edges.
			*/
            cvCanny( img, temp, 0, 50, 3 );
            cvIntegral( temp, sumcanny );
        }
    
        if( (unsigned)split_stage >= (unsigned)cascade->count ||
            cascade->hid_cascade->is_tree )
        {
            split_stage = cascade->count;
            npass = 1;
        }

		int step = 0;

		double cvSetImagesTime = 0.;
		double passTime = 0.;
		double cvRunImagesTime = 0.;
		double ixTime = 0.;
//#define FOR_CYCLE				// измеряет суммарное время исполнения цикла for (factor = 1...)
//#define cvSetImage			// измеряет время выполнения cvSetImagesForHaarClassifierCascade для каждого вызова функции
//#define PASS					// измеряет суммарное время исполнения цикла for (pass = 0...)
//#define CVRUNTIME				// измеряет время выполнения cvRunHaarClassifierCascade для каждого вызова функции
//#define IXTIME				// измеряет суммарное время исполнения цикла for (_ix = 0...)
//#define FOR_FACTOR_ITERATION	// измеряет время выполнения каждого цикла for (factor = 1...) в отдельности для каждой итерации
//#define SaveFile				// сохраняет массив mask_row в файл *.pgm для каждой итерации цикла for (factor = 1... ) и цикла pass
//#define MASK_ROW_WORK			// измеряет сколько произошло операций обращения в массив mask_row при pass = 0 и сохр. результир. прямоугольника при pass = 1
//#define cvRunCount				// измеряет сколько раз вызывается функция cvRunHaarClassifierCascade при pass = 0 и при pass = 1

#ifdef FOR_CYCLE
		double startTimeFactor;
		startTimeFactor = (double)cvGetTickCount();
#endif
		int ss = 0;
		int _ss = 0;
		double startTimeFactorForIteration;
		for( factor = 1; factor*cascade->orig_window_size.width < img->cols - 10 &&
                         factor*cascade->orig_window_size.height < img->rows - 10;
             factor *= scale_factor )
        {
#ifdef FOR_FACTOR_ITERATION
			startTimeFactorForIteration = (double)cvGetTickCount();
#endif
			// изм. 1
			iterCounter[step * 7 + 0]++;
            const double ystep = MAX( 2, factor );
			// вычисляем размер окна (?)
            CvSize win_size = { cvRound( cascade->orig_window_size.width * factor ),
                                cvRound( cascade->orig_window_size.height * factor )};
            // 
			CvRect equ_rect = { 0, 0, 0, 0 };
            int *p0 = 0, *p1 = 0, *p2 = 0, *p3 = 0;
            int *pq0 = 0, *pq1 = 0, *pq2 = 0, *pq3 = 0;
            int pass, stage_offset = 0;
			// число итераций по оси Oy
            int stop_height = cvRound((img->rows - win_size.height) / ystep);
			// увелииваем размер окна до тех пор, пока не достигнем минимального указанного размера при входе в ф-цию
            if( win_size.width < min_size.width || win_size.height < min_size.height )
			{
				// изм. 2
				iterCounter[step * 7 + 1]++;
				step++;
                continue;
			}
#ifdef MASK_ROW_WORK
			int pass0work;
			int pass1work;
#endif

#ifdef cvRunCount
			int _pass0work;
			int _pass1work;
#endif

			

/*
Assigns images to the hidden cascade
void cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                          const CvArr* sum, const CvArr* sqsum,
                                          const CvArr* tilted_sum, double scale );

cascade		-	каскад классификатора Хаара, созданный cvCreateHidHaarClassifierCascade. 
sum			-	Integral (sum) single-channel image of 32-bit integer format. This image as well as the two subsequent images are used for fast feature evaluation and brightness/contrast normalization. They all can be retrieved from input 8-bit or floating point single-channel image using The function cvIntegral. 
sqsum		-	Square sum single-channel image of 64-bit floating-point format. 
tilted_sum	-	Tilted sum single-channel image of 32-bit integer format. 
scale		-	Window scale for the cascade. If scale=1, original window size is used (objects of that size are searched) - the same size as specified in cvLoadHaarClassifierCascade (24x24 in case of "<default_face_cascade>"), if scale=2, a two times larger window is used (48x48 in case of default face cascade). While this will speed-up search about four times, faces smaller than 48x48 cannot be detected. 

The function cvSetImagesForHaarClassifierCascade assigns images and/or window scale to the hidden classifier cascade. If image pointers are NULL, the previously set images are used further (i.e. NULLs mean "do not change images"). Scale parameter has no such a "protection" value, but the previous value can be retrieved by cvGetHaarClassifierCascadeScale function and reused again. The function is used to prepare cascade for detecting object of the particular size in the particular image. The function is called internally by cvHaarDetectObjects, but it can be called by user if there is a need in using lower-level function cvRunHaarClassifierCascade.
*/
#ifdef cvSetImage
			cvSetImagesTime = (double)cvGetTickCount();
#endif
            cvSetImagesForHaarClassifierCascade( cascade, sum, sqsum, tilted, factor );
#ifdef cvSetImage
			cvSetImagesTime = (double)cvGetTickCount() - cvSetImagesTime;
			printf("cvSetImagesForHaarClassifierCascade = \t%g ms\n", cvSetImagesTime /( (double)cvGetTickFrequency() * 1000. ));
#endif
            cvZero( temp );
			cvZero( _temp_1 );
			cvZero( _temp_2 );
			cvZero( _temp_3 );

            if( do_canny_pruning )
            {	// ??????????????
                equ_rect.x = cvRound(win_size.width*0.15);
                equ_rect.y = cvRound(win_size.height*0.15);
                equ_rect.width = cvRound(win_size.width*0.7);
                equ_rect.height = cvRound(win_size.height*0.7);

				// вычисляем границу данных в соотв. с заданным окном
				// Canny = осторожный, осмотрительный
                p0 = (int*)(sumcanny->data.ptr + equ_rect.y*sumcanny->step) + equ_rect.x;
                p1 = (int*)(sumcanny->data.ptr + equ_rect.y*sumcanny->step)
                            + equ_rect.x + equ_rect.width;
                p2 = (int*)(sumcanny->data.ptr + (equ_rect.y + equ_rect.height)*sumcanny->step) + equ_rect.x;
                p3 = (int*)(sumcanny->data.ptr + (equ_rect.y + equ_rect.height)*sumcanny->step)
                            + equ_rect.x + equ_rect.width;

				// 
                pq0 = (int*)(sum->data.ptr + equ_rect.y*sum->step) + equ_rect.x;
                pq1 = (int*)(sum->data.ptr + equ_rect.y*sum->step)
                            + equ_rect.x + equ_rect.width;
                pq2 = (int*)(sum->data.ptr + (equ_rect.y + equ_rect.height)*sum->step) + equ_rect.x;
                pq3 = (int*)(sum->data.ptr + (equ_rect.y + equ_rect.height)*sum->step)
                            + equ_rect.x + equ_rect.width;
            }	// if( do_canny_pruning )

			static bool isPass0 = true;
            cascade->hid_cascade->count = split_stage;
#ifdef PASS
			passTime = (double)cvGetTickCount();
#endif
#ifdef MASK_ROW_WORK
	pass1work = pass0work = 0;
#endif

#ifdef cvRunCount
	_pass0work = _pass1work = 0;
#endif
            for( pass = 0; pass < npass; pass++ )
            {
				// изм. 3
				iterCounter[step * 7 + 2]++;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(max_threads), schedule(dynamic)
#endif
				// цикл по Oy
				bool _isPass0 = false;
                for( int _iy = 0; _iy < stop_height; _iy++ )
                {
					// изм. 4
					iterCounter[step * 7 + 3]++;
					// текущий индекс по оси Oy
                    int iy = cvRound(_iy*ystep);
                    int _ix, _xstep = 1;
					// число итераций по оси Ox
                    int stop_width = cvRound((img->cols - win_size.width) / ystep);
					// указатель на массив данных с учетом текущих координат
                    uchar* mask_row = temp->data.ptr + temp->step * iy;
					uchar* mask_row_1 = _temp_1->data.ptr + _temp_1->step * iy;
					uchar* mask_row_2 = _temp_2->data.ptr + _temp_2->step * iy;
					uchar* mask_row_3 = _temp_3->data.ptr + _temp_3->step * iy;
#ifdef IXTIME
	ixTime = (double)cvGetTickCount();
#endif
					// цикл по оси Ox
                    for( _ix = 0; _ix < stop_width; _ix += _xstep )
                    {
						// изм. 5
						iterCounter[step * 7 + 4]++;
						// текущий индекс по оси Ox
                        int ix = cvRound(_ix*ystep); // it really should be ystep
                    
                        if( pass == 0 )
                        {
							_isPass0 = true;
							// изм. 6
							iterCounter[step * 7 + 5]++;
                            int result;
							mask_row_2[ix] = ix;
							mask_row_3[ix] = _ix;
//							printf("ix = %d. _ix = %d. ", ix, _ix);
                            _xstep = 2;

                            if( do_canny_pruning )
                            {
                                int offset;
                                int s, sq;
								// вычисляем смещение
                                offset = iy*(sum->step/sizeof(p0[0])) + ix;
								// рассчитываем значение s и sq (см. "Emperical Analysis of Detection Cascades of Boosted..." п. 2.2, вычисление  RecSum)
                                s = p0[offset] - p1[offset] - p2[offset] + p3[offset];
                                sq = pq0[offset] - pq1[offset] - pq2[offset] + pq3[offset];
                                if( s < 100 || sq < 20 )
                                    continue;
                            }	// if( do_canny_pruning )
/*
int cvRunHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                CvPoint pt, int start_stage=0 );

cascade		-	классификатор каскада Хаара.
pt			-	верхний левый угол анализируемого региона 
start_stage	-	номер каскада, с которого начинать.

Запускает каскадный классификатор Хаара с указанной позиции.
Перед использованием integral images и соотв. масштаб (=> размер окна) д.б. уст. с пом. ф-ции cvSetImagesForHaarClassifierCascade
Возвр. значение >0, если анализируемый прямоугольник прошел все уровни (stage) классификатор, иначе <=0
*/
                    #ifdef CVRUNTIME
						cvRunImagesTime = (double)cvGetTickCount();
					#endif
						result = cvRunHaarClassifierCascade( cascade, cvPoint(ix,iy), 0 );
#ifdef cvRunCount
	_pass0work++;
#endif
					#ifdef CVRUNTIME
						cvRunImagesTime = (double)cvGetTickCount() - cvRunImagesTime;
						printf("cvRunHaarClassifierCascade with 0 offset = \t%g ms\n", cvRunImagesTime /( (double)cvGetTickFrequency() * 1000. ));
					#endif
							mask_row_1[ix] = result;
//							printf(". result = %d\t", result);
                            if( result > 0 )
                            {
								// ???
                                if( pass < npass - 1 )
								{
                                    mask_row[ix] = 255;
#ifdef MASK_ROW_WORK
		pass0work++;
#endif
								}
                                else
                                {
									//
                                    CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
#ifndef _OPENMP
									//
                                    cvSeqPush( seq, &rect );
#else
                                    cvSeqPush( seq_thread[omp_get_thread_num()], &rect );
#endif
                                }
                            }	// if( result > 0 )
							//
                            if( result < 0 )
                                _xstep = 1;
                        } //	if( pass == 0 )
                        else if( mask_row[ix] )
                        {
							// изм. 7
							iterCounter[step * 7 + 6]++;
					#ifdef CVRUNTIME
						cvRunImagesTime = (double)cvGetTickCount();
					#endif
                            int result = cvRunHaarClassifierCascade( cascade, cvPoint(ix,iy),
                                                                     stage_offset );
#ifdef cvRunCount
	_pass1work++;
#endif
					#ifdef CVRUNTIME
						cvRunImagesTime = (double)cvGetTickCount() - cvRunImagesTime;
						printf("cvRunHaarClassifierCascade with stage_offset offset = \t%g ms\n", cvRunImagesTime /( (double)cvGetTickFrequency() * 1000. ));
					#endif
							//
                            if( result > 0 )
                            {
								//
                                if( pass == npass - 1 )
                                {
									// создаем прямоугольную область с тек. коорд. (ix,iy) и размером окна. Что-то нашли (?)
                                    CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
#ifdef MASK_ROW_WORK
		pass1work++;
#endif
#ifndef _OPENMP
									// добавляем найденный элемент в стек
                                    cvSeqPush( seq, &rect );
#else
                                    cvSeqPush( seq_thread[omp_get_thread_num()], &rect );
#endif
                                }
                            }
                            else
								//
                                mask_row[ix] = 0;
                        }	// if( mask_row[ix] )
                    }	// for( _ix = 0; _ix < stop_width; _ix += _xstep )
#ifdef IXTIME
	ixTime = (double)cvGetTickCount() - ixTime;
	printf("ix cycle time = \t%g ms\n", ixTime /( (double)cvGetTickFrequency() * 1000. ));
#endif
//					printf("\n");
				}	// for( int _iy = 0; _iy < stop_height; _iy++ )
/*				char fileIn_[150];
				char temp__[150];
				//_ss++;
				if (isPass0)
				{
					strcpy_s( fileIn_, "resultArray" );
					_itoa_s( _ss, temp__, 150, 10 );
					strcat_s( fileIn_, temp__);
					strcat_s( fileIn_, "Pass0");
					strcat_s( fileIn_, ".txt" );
					FILE* _fpout = fopen(fileIn_,"wb");
					int s1 = 0, s2 = 0;
					for (s1 = 0; s1 < _temp_1->height; s1++)
					{
						uchar* mask_row_1 = _temp_1->data.ptr + _temp_1->step * s1;
						uchar* mask_row_2 = _temp_2->data.ptr + _temp_2->step * s1;
						uchar* mask_row_3 = _temp_3->data.ptr + _temp_3->step * s1;
						for (s2 = 0; s2 < _temp_1->width; s2++)
							if (mask_row_3[s2] != 0 || ( mask_row_3[s2] == 0 && mask_row_1[s2] != 0 ) )
								fprintf(_fpout, "result = %d. ix = %d. _ix = %d\t", mask_row_1[s2], mask_row_2[s2], mask_row_3[s2]);
						fprintf(_fpout, "\n");
					}
					_ss++;
					isPass0 = false;
					fclose(_fpout);
				}*/
#ifdef SaveFile
				char fileIn[150];
				char temp_[150];
				if (isPass0)
				{
					strcpy_s( fileIn, "Image" );
					_itoa_s( ss, temp_, 150, 10 );
					strcat_s( fileIn, temp_);
					strcat_s( fileIn, "Pass0");
					strcat_s( fileIn, ".pgm" );
					FILE* fpout = fopen(fileIn,"wb");
					save_pgm(fpout, temp->width, temp->height, temp->data.ptr);
					//ss++;
					isPass0 = false;
					fclose(fpout);
				}
				else
				{
					strcpy_s( fileIn, "Image" );
					_itoa_s( ss, temp_, 150, 10 );
					strcat_s( fileIn, temp_);
					strcat_s( fileIn, "Pass1");
					strcat_s( fileIn, ".pgm" );
					FILE* fpout = fopen(fileIn,"wb");
					save_pgm(fpout, temp->width, temp->height, temp->data.ptr);
					ss++;
					isPass0 = true;
					fclose(fpout);
				}
#endif
				//
                stage_offset = cascade->hid_cascade->count;
				//
                cascade->hid_cascade->count = cascade->count;
            }	// for( pass = 0; pass < npass; pass++ )

#ifdef MASK_ROW_WORK
		printf("factor = %g. \n\tmask_row work with pass 0 = %d\n\tmask_row work with pass 1 = %d\n", factor, pass0work, pass1work);
#endif

#ifdef cvRunCount
		printf("factor = %g. \n\tcvRun count with pass 0 = %d\n\tcvRun count with pass 1 = %d\n", factor, _pass0work, _pass1work);
#endif		

#ifdef PASS
			passTime = (double)cvGetTickCount() - passTime;
			printf("npass cycle = \t%g ms\n", passTime /( (double)cvGetTickFrequency() * 1000. ));
#endif
			step++;
#ifdef FOR_FACTOR_ITERATION
			startTimeFactorForIteration = (double)cvGetTickCount() - startTimeFactorForIteration;
			printf("factor iteration. factor = %g. Time = \t%g ms\n", factor, startTimeFactorForIteration /( (double)cvGetTickFrequency() * 1000. ));
#endif
		}	// for( factor = 1;
#ifdef FOR_CYCLE
		startTimeFactor = (double)cvGetTickCount() - startTimeFactor;
		printf("for factor cycle time = \t%g ms\n", startTimeFactor /( (double)cvGetTickFrequency() * 1000. ));
#endif
	}	// else // if( flags & CV_HAAR_SCALE_IMAGE )

#ifdef _OPENMP
	// gather the results
	for( i = 0; i < max_threads; i++ )
	{
		CvSeq* s = seq_thread[i];
        int j, total = s->total;
        CvSeqBlock* b = s->first;
        for( j = 0; j < total; j += b->count, b = b->next )
            cvSeqPushMulti( seq, b->data, b->count );
	}
#endif

	// если мин. число из ближайшего окружения не ноль (передается из main)
    if( min_neighbors != 0 )
    {
        // group retrieved rectangles in order to filter out noise
		// расщепляет последовательность seq на одну или более эквивалентных классов по спец. критерию
/*
int cvSeqPartition(const CvSeq* seq, CvMemStorage* storage, CvSeq** labels, CvCmpFunc is_equal, void* userdata)

seq			– The sequence to partition
storage		– The storage block to store the sequence of equivalency classes. If it is NULL, the function uses seq->storage for output labels
labels		– Ouput parameter. Double pointer to the sequence of 0-based labels of input sequence elements
is_equal	– The relation function that should return non-zero if the two particular sequence elements are from the same class, and zero otherwise. The partitioning algorithm uses transitive closure of the relation function as an equivalency critria
userdata	– Pointer that is transparently passed to the is_equal function
*/
        int ncomp = cvSeqPartition( seq, 0, &idx_seq, is_equal, 0 );
		// выделяем память для CvAvgComp (описывает Rect и neighbours)
        CV_CALL( comps = (CvAvgComp*)cvAlloc( (ncomp+1)*sizeof(comps[0])));
        memset( comps, 0, (ncomp+1)*sizeof(comps[0]));

        // count number of neighbors
        for( i = 0; i < seq->total; i++ )
        {
			// изм. 8
			iterCounter[40*7 + 1]++;
			// извлекаем i-ый элемент и последовательности
            CvRect r1 = *(CvRect*)cvGetSeqElem( seq, i );
            int idx = *(int*)cvGetSeqElem( idx_seq, i );
			// проверяем выполнение условия
            assert( (unsigned)idx < (unsigned)ncomp );

			// наращиваем границы кандидата
            comps[idx].neighbors++;
             
            comps[idx].rect.x += r1.x;
            comps[idx].rect.y += r1.y;
            comps[idx].rect.width += r1.width;
            comps[idx].rect.height += r1.height;
        }

        // calculate average bounding box
		// вычисляем среднее значение граничного "ящика" (окна (?) )
        for( i = 0; i < ncomp; i++ )
        {
			// изм. 9
			iterCounter[40*7 + 2]++;
            int n = comps[i].neighbors;
			// сравниваем с мин. значение соседей
            if( n >= min_neighbors )
            {
				// нашли область, удовлетворяющую условиям
                CvAvgComp comp;
                comp.rect.x = (comps[i].rect.x*2 + n)/(2*n);
                comp.rect.y = (comps[i].rect.y*2 + n)/(2*n);
                comp.rect.width = (comps[i].rect.width*2 + n)/(2*n);
                comp.rect.height = (comps[i].rect.height*2 + n)/(2*n);
                comp.neighbors = comps[i].neighbors;

				// сохр. в последовательность seq2
                cvSeqPush( seq2, &comp );
            }
        }

        // filter out small face rectangles inside large face rectangles
		//
        for( i = 0; i < seq2->total; i++ )
        {
			// изм. 10
			iterCounter[40*7 + 3]++;
			// извлекаем очередного кандидата
            CvAvgComp r1 = *(CvAvgComp*)cvGetSeqElem( seq2, i );
            int j, flag = 1;
			// пытаемся найти области, входящие в рассматриваемую в тек. момент
            for( j = 0; j < seq2->total; j++ )
            {
				// извлекаем очередного кандидата и рассчитываем предполагаемую допустимую дистанцию
                CvAvgComp r2 = *(CvAvgComp*)cvGetSeqElem( seq2, j );
                int distance = cvRound( r2.rect.width * 0.2 );
            
				// r1 - входит в объект r2 ?
                if( i != j &&
                    r1.rect.x >= r2.rect.x - distance &&
                    r1.rect.y >= r2.rect.y - distance &&
                    r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
                    r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
                    (r2.neighbors > MAX( 3, r1.neighbors ) || r1.neighbors < 3) )
                {	// да. Флаг сбрасываем
                    flag = 0;
                    break;
                }
            }

            if( flag )
            {
				// если флаг установлен, то последовательность r1 уникальна. Следовательно, передаем в вых. последовательность
                cvSeqPush( result_seq, &r1 );
                /* cvSeqPush( result_seq, &r1.rect ); */
            }
        }
    }

    __END__;

#ifdef _OPENMP
	for( i = 0; i < max_threads; i++ )
	{
		if( seq_thread[i] )
            cvReleaseMemStorage( &seq_thread[i]->storage );
	}
#endif

	// освобождаем всю память
    cvReleaseMemStorage( &temp_storage );
    cvReleaseMat( &sum );
    cvReleaseMat( &sqsum );
    cvReleaseMat( &tilted );
    cvReleaseMat( &temp );
    cvReleaseMat( &sumcanny );
    cvReleaseMat( &norm_img );
    cvReleaseMat( &img_small );
    cvFree( &comps );

    return result_seq;
}

static CvHaarClassifierCascade*
icvLoadCascadeCART( const char** input_cascade, int n, CvSize orig_window_size )
{
    int i;
    CvHaarClassifierCascade* cascade = icvCreateHaarClassifierCascade(n);
    cascade->orig_window_size = orig_window_size;

    for( i = 0; i < n; i++ )
    {
        int j, count, l;
        float threshold = 0;
        const char* stage = input_cascade[i];
        int dl = 0;

        /* tree links */
        int parent = -1;
        int next = -1;

        sscanf( stage, "%d%n", &count, &dl );
        stage += dl;
        
        assert( count > 0 );
        cascade->stage_classifier[i].count = count;
        cascade->stage_classifier[i].classifier =
            (CvHaarClassifier*)cvAlloc( count*sizeof(cascade->stage_classifier[i].classifier[0]));

        for( j = 0; j < count; j++ )
        {
            CvHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
            int k, rects = 0;
            char str[100];
            
            sscanf( stage, "%d%n", &classifier->count, &dl );
            stage += dl;

            classifier->haar_feature = (CvHaarFeature*) cvAlloc( 
                classifier->count * ( sizeof( *classifier->haar_feature ) +
                                      sizeof( *classifier->threshold ) +
                                      sizeof( *classifier->left ) +
                                      sizeof( *classifier->right ) ) +
                (classifier->count + 1) * sizeof( *classifier->alpha ) );
            classifier->threshold = (float*) (classifier->haar_feature+classifier->count);
            classifier->left = (int*) (classifier->threshold + classifier->count);
            classifier->right = (int*) (classifier->left + classifier->count);
            classifier->alpha = (float*) (classifier->right + classifier->count);
            
            for( l = 0; l < classifier->count; l++ )
            {
                sscanf( stage, "%d%n", &rects, &dl );
                stage += dl;

                assert( rects >= 2 && rects <= CV_HAAR_FEATURE_MAX );

                for( k = 0; k < rects; k++ )
                {
                    CvRect r;
                    int band = 0;
                    sscanf( stage, "%d%d%d%d%d%f%n",
                            &r.x, &r.y, &r.width, &r.height, &band,
                            &(classifier->haar_feature[l].rect[k].weight), &dl );
                    stage += dl;
                    classifier->haar_feature[l].rect[k].r = r;
                }
                sscanf( stage, "%s%n", str, &dl );
                stage += dl;
            
                classifier->haar_feature[l].tilted = strncmp( str, "tilted", 6 ) == 0;
            
                for( k = rects; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    memset( classifier->haar_feature[l].rect + k, 0,
                            sizeof(classifier->haar_feature[l].rect[k]) );
                }
                
                sscanf( stage, "%f%d%d%n", &(classifier->threshold[l]), 
                                       &(classifier->left[l]),
                                       &(classifier->right[l]), &dl );
                stage += dl;
            }
            for( l = 0; l <= classifier->count; l++ )
            {
                sscanf( stage, "%f%n", &(classifier->alpha[l]), &dl );
                stage += dl;
            }
        }
       
        sscanf( stage, "%f%n", &threshold, &dl );
        stage += dl;

        cascade->stage_classifier[i].threshold = threshold;

        /* load tree links */
        if( sscanf( stage, "%d%d%n", &parent, &next, &dl ) != 2 )
        {
            parent = i - 1;
            next = -1;
        }
        stage += dl;

        cascade->stage_classifier[i].parent = parent;
        cascade->stage_classifier[i].next = next;
        cascade->stage_classifier[i].child = -1;

        if( parent != -1 && cascade->stage_classifier[parent].child == -1 )
        {
            cascade->stage_classifier[parent].child = i;
        }
    }

    return cascade;
}

#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif

CV_IMPL CvHaarClassifierCascade*
cvLoadHaarClassifierCascade( const char* directory, CvSize orig_window_size )
{
    const char** input_cascade = 0; 
    CvHaarClassifierCascade *cascade = 0;

    CV_FUNCNAME( "cvLoadHaarClassifierCascade" );

    __BEGIN__;

    int i, n;
    const char* slash;
    char name[_MAX_PATH];
    int size = 0;
    char* ptr = 0;

    if( !directory )
        CV_ERROR( CV_StsNullPtr, "Null path is passed" );

    n = (int)strlen(directory)-1;
    slash = directory[n] == '\\' || directory[n] == '/' ? "" : "/";

    /* try to read the classifier from directory */
    for( n = 0; ; n++ )
    {
        sprintf( name, "%s%s%d/AdaBoostCARTHaarClassifier.txt", directory, slash, n );
        FILE* f = fopen( name, "rb" );
        if( !f )
            break;
        fseek( f, 0, SEEK_END );
        size += ftell( f ) + 1;
        fclose(f);
    }

    if( n == 0 && slash[0] )
    {
        CV_CALL( cascade = (CvHaarClassifierCascade*)cvLoad( directory ));
        EXIT;
    }
    else if( n == 0 )
        CV_ERROR( CV_StsBadArg, "Invalid path" );
    
    size += (n+1)*sizeof(char*);
    CV_CALL( input_cascade = (const char**)cvAlloc( size ));
    ptr = (char*)(input_cascade + n + 1);

    for( i = 0; i < n; i++ )
    {
        sprintf( name, "%s/%d/AdaBoostCARTHaarClassifier.txt", directory, i );
        FILE* f = fopen( name, "rb" );
        if( !f )
            CV_ERROR( CV_StsError, "" );
        fseek( f, 0, SEEK_END );
        size = ftell( f );
        fseek( f, 0, SEEK_SET );
        fread( ptr, 1, size, f );
        fclose(f);
        input_cascade[i] = ptr;
        ptr += size;
        *ptr++ = '\0';
    }

    input_cascade[n] = 0;
    cascade = icvLoadCascadeCART( input_cascade, n, orig_window_size );

    __END__;

    if( input_cascade )
        cvFree( &input_cascade );

    if( cvGetErrStatus() < 0 )
        cvReleaseHaarClassifierCascade( &cascade );

    return cascade;
}


CV_IMPL void
cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** _cascade )
{
    if( _cascade && *_cascade )
    {
        int i, j;
        CvHaarClassifierCascade* cascade = *_cascade;

        for( i = 0; i < cascade->count; i++ )
        {
            for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                cvFree( &cascade->stage_classifier[i].classifier[j].haar_feature );
            cvFree( &cascade->stage_classifier[i].classifier );
        }
        icvReleaseHidHaarClassifierCascade( &cascade->hid_cascade );
        cvFree( _cascade );
    }
}


/****************************************************************************************\
*                                  Persistence functions                                 *
\****************************************************************************************/

/* field names */

#define ICV_HAAR_SIZE_NAME            "size"
#define ICV_HAAR_STAGES_NAME          "stages"
#define ICV_HAAR_TREES_NAME             "trees"
#define ICV_HAAR_FEATURE_NAME             "feature"
#define ICV_HAAR_RECTS_NAME                 "rects"
#define ICV_HAAR_TILTED_NAME                "tilted"
#define ICV_HAAR_THRESHOLD_NAME           "threshold"
#define ICV_HAAR_LEFT_NODE_NAME           "left_node"
#define ICV_HAAR_LEFT_VAL_NAME            "left_val"
#define ICV_HAAR_RIGHT_NODE_NAME          "right_node"
#define ICV_HAAR_RIGHT_VAL_NAME           "right_val"
#define ICV_HAAR_STAGE_THRESHOLD_NAME   "stage_threshold"
#define ICV_HAAR_PARENT_NAME            "parent"
#define ICV_HAAR_NEXT_NAME              "next"

static int
icvIsHaarClassifier( const void* struct_ptr )
{
    return CV_IS_HAAR_CLASSIFIER( struct_ptr );
}

static void*
icvReadHaarClassifier( CvFileStorage* fs, CvFileNode* node )
{
    CvHaarClassifierCascade* cascade = NULL;

    CV_FUNCNAME( "cvReadHaarClassifier" );

    __BEGIN__;

    char buf[256];
    CvFileNode* seq_fn = NULL; /* sequence */
    CvFileNode* fn = NULL;
    CvFileNode* stages_fn = NULL;
    CvSeqReader stages_reader;
    int n;
    int i, j, k, l;
    int parent, next;

    CV_CALL( stages_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_STAGES_NAME ) );
    if( !stages_fn || !CV_NODE_IS_SEQ( stages_fn->tag) )
        CV_ERROR( CV_StsError, "Invalid stages node" );

    n = stages_fn->data.seq->total;
    CV_CALL( cascade = icvCreateHaarClassifierCascade(n) );

    /* read size */
    CV_CALL( seq_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_SIZE_NAME ) );
    if( !seq_fn || !CV_NODE_IS_SEQ( seq_fn->tag ) || seq_fn->data.seq->total != 2 )
        CV_ERROR( CV_StsError, "size node is not a valid sequence." );
    CV_CALL( fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 0 ) );
    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
        CV_ERROR( CV_StsError, "Invalid size node: width must be positive integer" );
    cascade->orig_window_size.width = fn->data.i;
    CV_CALL( fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 1 ) );
    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
        CV_ERROR( CV_StsError, "Invalid size node: height must be positive integer" );
    cascade->orig_window_size.height = fn->data.i;

    CV_CALL( cvStartReadSeq( stages_fn->data.seq, &stages_reader ) );
    for( i = 0; i < n; ++i )
    {
        CvFileNode* stage_fn;
        CvFileNode* trees_fn;
        CvSeqReader trees_reader;

        stage_fn = (CvFileNode*) stages_reader.ptr;
        if( !CV_NODE_IS_MAP( stage_fn->tag ) )
        {
            sprintf( buf, "Invalid stage %d", i );
            CV_ERROR( CV_StsError, buf );
        }

        CV_CALL( trees_fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_TREES_NAME ) );
        if( !trees_fn || !CV_NODE_IS_SEQ( trees_fn->tag )
            || trees_fn->data.seq->total <= 0 )
        {
            sprintf( buf, "Trees node is not a valid sequence. (stage %d)", i );
            CV_ERROR( CV_StsError, buf );
        }

        CV_CALL( cascade->stage_classifier[i].classifier =
            (CvHaarClassifier*) cvAlloc( trees_fn->data.seq->total
                * sizeof( cascade->stage_classifier[i].classifier[0] ) ) );
        for( j = 0; j < trees_fn->data.seq->total; ++j )
        {
            cascade->stage_classifier[i].classifier[j].haar_feature = NULL;
        }
        cascade->stage_classifier[i].count = trees_fn->data.seq->total;

        CV_CALL( cvStartReadSeq( trees_fn->data.seq, &trees_reader ) );
        for( j = 0; j < trees_fn->data.seq->total; ++j )
        {
            CvFileNode* tree_fn;
            CvSeqReader tree_reader;
            CvHaarClassifier* classifier;
            int last_idx;

            classifier = &cascade->stage_classifier[i].classifier[j];
            tree_fn = (CvFileNode*) trees_reader.ptr;
            if( !CV_NODE_IS_SEQ( tree_fn->tag ) || tree_fn->data.seq->total <= 0 )
            {
                sprintf( buf, "Tree node is not a valid sequence."
                         " (stage %d, tree %d)", i, j );
                CV_ERROR( CV_StsError, buf );
            }

            classifier->count = tree_fn->data.seq->total;
            CV_CALL( classifier->haar_feature = (CvHaarFeature*) cvAlloc( 
                classifier->count * ( sizeof( *classifier->haar_feature ) +
                                      sizeof( *classifier->threshold ) +
                                      sizeof( *classifier->left ) +
                                      sizeof( *classifier->right ) ) +
                (classifier->count + 1) * sizeof( *classifier->alpha ) ) );
            classifier->threshold = (float*) (classifier->haar_feature+classifier->count);
            classifier->left = (int*) (classifier->threshold + classifier->count);
            classifier->right = (int*) (classifier->left + classifier->count);
            classifier->alpha = (float*) (classifier->right + classifier->count);

            CV_CALL( cvStartReadSeq( tree_fn->data.seq, &tree_reader ) );
            for( k = 0, last_idx = 0; k < tree_fn->data.seq->total; ++k )
            {
                CvFileNode* node_fn;
                CvFileNode* feature_fn;
                CvFileNode* rects_fn;
                CvSeqReader rects_reader;

                node_fn = (CvFileNode*) tree_reader.ptr;
                if( !CV_NODE_IS_MAP( node_fn->tag ) )
                {
                    sprintf( buf, "Tree node %d is not a valid map. (stage %d, tree %d)",
                             k, i, j );
                    CV_ERROR( CV_StsError, buf );
                }
                CV_CALL( feature_fn = cvGetFileNodeByName( fs, node_fn,
                    ICV_HAAR_FEATURE_NAME ) );
                if( !feature_fn || !CV_NODE_IS_MAP( feature_fn->tag ) )
                {
                    sprintf( buf, "Feature node is not a valid map. "
                             "(stage %d, tree %d, node %d)", i, j, k );
                    CV_ERROR( CV_StsError, buf );
                }
                CV_CALL( rects_fn = cvGetFileNodeByName( fs, feature_fn,
                    ICV_HAAR_RECTS_NAME ) );
                if( !rects_fn || !CV_NODE_IS_SEQ( rects_fn->tag )
                    || rects_fn->data.seq->total < 1
                    || rects_fn->data.seq->total > CV_HAAR_FEATURE_MAX )
                {
                    sprintf( buf, "Rects node is not a valid sequence. "
                             "(stage %d, tree %d, node %d)", i, j, k );
                    CV_ERROR( CV_StsError, buf );
                }
                CV_CALL( cvStartReadSeq( rects_fn->data.seq, &rects_reader ) );
                for( l = 0; l < rects_fn->data.seq->total; ++l )
                {
                    CvFileNode* rect_fn;
                    CvRect r;

                    rect_fn = (CvFileNode*) rects_reader.ptr;
                    if( !CV_NODE_IS_SEQ( rect_fn->tag ) || rect_fn->data.seq->total != 5 )
                    {
                        sprintf( buf, "Rect %d is not a valid sequence. "
                                 "(stage %d, tree %d, node %d)", l, i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    
                    fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 0 );
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
                    {
                        sprintf( buf, "x coordinate must be non-negative integer. "
                                 "(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
                        CV_ERROR( CV_StsError, buf );
                    }
                    r.x = fn->data.i;
                    fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 1 );
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
                    {
                        sprintf( buf, "y coordinate must be non-negative integer. "
                                 "(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
                        CV_ERROR( CV_StsError, buf );
                    }
                    r.y = fn->data.i;
                    fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 2 );
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
                        || r.x + fn->data.i > cascade->orig_window_size.width )
                    {
                        sprintf( buf, "width must be positive integer and "
                                 "(x + width) must not exceed window width. "
                                 "(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
                        CV_ERROR( CV_StsError, buf );
                    }
                    r.width = fn->data.i;
                    fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 3 );
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
                        || r.y + fn->data.i > cascade->orig_window_size.height )
                    {
                        sprintf( buf, "height must be positive integer and "
                                 "(y + height) must not exceed window height. "
                                 "(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
                        CV_ERROR( CV_StsError, buf );
                    }
                    r.height = fn->data.i;
                    fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 4 );
                    if( !CV_NODE_IS_REAL( fn->tag ) )
                    {
                        sprintf( buf, "weight must be real number. "
                                 "(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
                        CV_ERROR( CV_StsError, buf );
                    }

                    classifier->haar_feature[k].rect[l].weight = (float) fn->data.f;
                    classifier->haar_feature[k].rect[l].r = r;

                    CV_NEXT_SEQ_ELEM( sizeof( *rect_fn ), rects_reader );
                } /* for each rect */
                for( l = rects_fn->data.seq->total; l < CV_HAAR_FEATURE_MAX; ++l )
                {
                    classifier->haar_feature[k].rect[l].weight = 0;
                    classifier->haar_feature[k].rect[l].r = cvRect( 0, 0, 0, 0 );
                }

                CV_CALL( fn = cvGetFileNodeByName( fs, feature_fn, ICV_HAAR_TILTED_NAME));
                if( !fn || !CV_NODE_IS_INT( fn->tag ) )
                {
                    sprintf( buf, "tilted must be 0 or 1. "
                             "(stage %d, tree %d, node %d)", i, j, k );
                    CV_ERROR( CV_StsError, buf );
                }
                classifier->haar_feature[k].tilted = ( fn->data.i != 0 );
                CV_CALL( fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_THRESHOLD_NAME));
                if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
                {
                    sprintf( buf, "threshold must be real number. "
                             "(stage %d, tree %d, node %d)", i, j, k );
                    CV_ERROR( CV_StsError, buf );
                }
                classifier->threshold[k] = (float) fn->data.f;
                CV_CALL( fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_LEFT_NODE_NAME));
                if( fn )
                {
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
                        || fn->data.i >= tree_fn->data.seq->total )
                    {
                        sprintf( buf, "left node must be valid node number. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    /* left node */
                    classifier->left[k] = fn->data.i;
                }
                else
                {
                    CV_CALL( fn = cvGetFileNodeByName( fs, node_fn,
                        ICV_HAAR_LEFT_VAL_NAME ) );
                    if( !fn )
                    {
                        sprintf( buf, "left node or left value must be specified. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    if( !CV_NODE_IS_REAL( fn->tag ) )
                    {
                        sprintf( buf, "left value must be real number. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    /* left value */
                    if( last_idx >= classifier->count + 1 )
                    {
                        sprintf( buf, "Tree structure is broken: too many values. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    classifier->left[k] = -last_idx;
                    classifier->alpha[last_idx++] = (float) fn->data.f;
                }
                CV_CALL( fn = cvGetFileNodeByName( fs, node_fn,ICV_HAAR_RIGHT_NODE_NAME));
                if( fn )
                {
                    if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
                        || fn->data.i >= tree_fn->data.seq->total )
                    {
                        sprintf( buf, "right node must be valid node number. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    /* right node */
                    classifier->right[k] = fn->data.i;
                }
                else
                {
                    CV_CALL( fn = cvGetFileNodeByName( fs, node_fn,
                        ICV_HAAR_RIGHT_VAL_NAME ) );
                    if( !fn )
                    {
                        sprintf( buf, "right node or right value must be specified. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    if( !CV_NODE_IS_REAL( fn->tag ) )
                    {
                        sprintf( buf, "right value must be real number. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    /* right value */
                    if( last_idx >= classifier->count + 1 )
                    {
                        sprintf( buf, "Tree structure is broken: too many values. "
                                 "(stage %d, tree %d, node %d)", i, j, k );
                        CV_ERROR( CV_StsError, buf );
                    }
                    classifier->right[k] = -last_idx;
                    classifier->alpha[last_idx++] = (float) fn->data.f;
                }

                CV_NEXT_SEQ_ELEM( sizeof( *node_fn ), tree_reader );
            } /* for each node */
            if( last_idx != classifier->count + 1 )
            {
                sprintf( buf, "Tree structure is broken: too few values. "
                         "(stage %d, tree %d)", i, j );
                CV_ERROR( CV_StsError, buf );
            }

            CV_NEXT_SEQ_ELEM( sizeof( *tree_fn ), trees_reader );
        } /* for each tree */

        CV_CALL( fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_STAGE_THRESHOLD_NAME));
        if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
        {
            sprintf( buf, "stage threshold must be real number. (stage %d)", i );
            CV_ERROR( CV_StsError, buf );
        }
        cascade->stage_classifier[i].threshold = (float) fn->data.f;

        parent = i - 1;
        next = -1;

        CV_CALL( fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_PARENT_NAME ) );
        if( !fn || !CV_NODE_IS_INT( fn->tag )
            || fn->data.i < -1 || fn->data.i >= cascade->count )
        {
            sprintf( buf, "parent must be integer number. (stage %d)", i );
            CV_ERROR( CV_StsError, buf );
        }
        parent = fn->data.i;
        CV_CALL( fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_NEXT_NAME ) );
        if( !fn || !CV_NODE_IS_INT( fn->tag )
            || fn->data.i < -1 || fn->data.i >= cascade->count )
        {
            sprintf( buf, "next must be integer number. (stage %d)", i );
            CV_ERROR( CV_StsError, buf );
        }
        next = fn->data.i;

        cascade->stage_classifier[i].parent = parent;
        cascade->stage_classifier[i].next = next;
        cascade->stage_classifier[i].child = -1;

        if( parent != -1 && cascade->stage_classifier[parent].child == -1 )
        {
            cascade->stage_classifier[parent].child = i;
        }
        
        CV_NEXT_SEQ_ELEM( sizeof( *stage_fn ), stages_reader );
    } /* for each stage */

    __END__;

    if( cvGetErrStatus() < 0 )
    {
        cvReleaseHaarClassifierCascade( &cascade );
        cascade = NULL;
    }

    return cascade;
}

static void
icvWriteHaarClassifier( CvFileStorage* fs, const char* name, const void* struct_ptr,
                        CvAttrList attributes )
{
    CV_FUNCNAME( "cvWriteHaarClassifier" );

    __BEGIN__;

    int i, j, k, l;
    char buf[256];
    const CvHaarClassifierCascade* cascade = (const CvHaarClassifierCascade*) struct_ptr;

    /* TODO: parameters check */

    CV_CALL( cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_HAAR, attributes ) );
    
    CV_CALL( cvStartWriteStruct( fs, ICV_HAAR_SIZE_NAME, CV_NODE_SEQ | CV_NODE_FLOW ) );
    CV_CALL( cvWriteInt( fs, NULL, cascade->orig_window_size.width ) );
    CV_CALL( cvWriteInt( fs, NULL, cascade->orig_window_size.height ) );
    CV_CALL( cvEndWriteStruct( fs ) ); /* size */
    
    CV_CALL( cvStartWriteStruct( fs, ICV_HAAR_STAGES_NAME, CV_NODE_SEQ ) );    
    for( i = 0; i < cascade->count; ++i )
    {
        CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_MAP ) );
        sprintf( buf, "stage %d", i );
        CV_CALL( cvWriteComment( fs, buf, 1 ) );
        
        CV_CALL( cvStartWriteStruct( fs, ICV_HAAR_TREES_NAME, CV_NODE_SEQ ) );

        for( j = 0; j < cascade->stage_classifier[i].count; ++j )
        {
            CvHaarClassifier* tree = &cascade->stage_classifier[i].classifier[j];

            CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_SEQ ) );
            sprintf( buf, "tree %d", j );
            CV_CALL( cvWriteComment( fs, buf, 1 ) );

            for( k = 0; k < tree->count; ++k )
            {
                CvHaarFeature* feature = &tree->haar_feature[k];

                CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_MAP ) );
                if( k )
                {
                    sprintf( buf, "node %d", k );
                }
                else
                {
                    sprintf( buf, "root node" );
                }
                CV_CALL( cvWriteComment( fs, buf, 1 ) );

                CV_CALL( cvStartWriteStruct( fs, ICV_HAAR_FEATURE_NAME, CV_NODE_MAP ) );
                
                CV_CALL( cvStartWriteStruct( fs, ICV_HAAR_RECTS_NAME, CV_NODE_SEQ ) );
                for( l = 0; l < CV_HAAR_FEATURE_MAX && feature->rect[l].r.width != 0; ++l )
                {
                    CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_SEQ | CV_NODE_FLOW ) );
                    CV_CALL( cvWriteInt(  fs, NULL, feature->rect[l].r.x ) );
                    CV_CALL( cvWriteInt(  fs, NULL, feature->rect[l].r.y ) );
                    CV_CALL( cvWriteInt(  fs, NULL, feature->rect[l].r.width ) );
                    CV_CALL( cvWriteInt(  fs, NULL, feature->rect[l].r.height ) );
                    CV_CALL( cvWriteReal( fs, NULL, feature->rect[l].weight ) );
                    CV_CALL( cvEndWriteStruct( fs ) ); /* rect */
                }
                CV_CALL( cvEndWriteStruct( fs ) ); /* rects */
                CV_CALL( cvWriteInt( fs, ICV_HAAR_TILTED_NAME, feature->tilted ) );
                CV_CALL( cvEndWriteStruct( fs ) ); /* feature */
                
                CV_CALL( cvWriteReal( fs, ICV_HAAR_THRESHOLD_NAME, tree->threshold[k]) );

                if( tree->left[k] > 0 )
                {
                    CV_CALL( cvWriteInt( fs, ICV_HAAR_LEFT_NODE_NAME, tree->left[k] ) );
                }
                else
                {
                    CV_CALL( cvWriteReal( fs, ICV_HAAR_LEFT_VAL_NAME,
                        tree->alpha[-tree->left[k]] ) );
                }

                if( tree->right[k] > 0 )
                {
                    CV_CALL( cvWriteInt( fs, ICV_HAAR_RIGHT_NODE_NAME, tree->right[k] ) );
                }
                else
                {
                    CV_CALL( cvWriteReal( fs, ICV_HAAR_RIGHT_VAL_NAME,
                        tree->alpha[-tree->right[k]] ) );
                }

                CV_CALL( cvEndWriteStruct( fs ) ); /* split */
            }

            CV_CALL( cvEndWriteStruct( fs ) ); /* tree */
        }

        CV_CALL( cvEndWriteStruct( fs ) ); /* trees */

        CV_CALL( cvWriteReal( fs, ICV_HAAR_STAGE_THRESHOLD_NAME,
                              cascade->stage_classifier[i].threshold) );

        CV_CALL( cvWriteInt( fs, ICV_HAAR_PARENT_NAME,
                              cascade->stage_classifier[i].parent ) );
        CV_CALL( cvWriteInt( fs, ICV_HAAR_NEXT_NAME,
                              cascade->stage_classifier[i].next ) );

        CV_CALL( cvEndWriteStruct( fs ) ); /* stage */
    } /* for each stage */
    
    CV_CALL( cvEndWriteStruct( fs ) ); /* stages */
    CV_CALL( cvEndWriteStruct( fs ) ); /* root */

    __END__;
}

static void*
icvCloneHaarClassifier( const void* struct_ptr )
{
    CvHaarClassifierCascade* cascade = NULL;

    CV_FUNCNAME( "cvCloneHaarClassifier" );

    __BEGIN__;

    int i, j, k, n;
    const CvHaarClassifierCascade* cascade_src =
        (const CvHaarClassifierCascade*) struct_ptr;

    n = cascade_src->count;
    CV_CALL( cascade = icvCreateHaarClassifierCascade(n) );
    cascade->orig_window_size = cascade_src->orig_window_size;

    for( i = 0; i < n; ++i )
    {
        cascade->stage_classifier[i].parent = cascade_src->stage_classifier[i].parent;
        cascade->stage_classifier[i].next = cascade_src->stage_classifier[i].next;
        cascade->stage_classifier[i].child = cascade_src->stage_classifier[i].child;
        cascade->stage_classifier[i].threshold = cascade_src->stage_classifier[i].threshold;

        cascade->stage_classifier[i].count = 0;
        CV_CALL( cascade->stage_classifier[i].classifier =
            (CvHaarClassifier*) cvAlloc( cascade_src->stage_classifier[i].count
                * sizeof( cascade->stage_classifier[i].classifier[0] ) ) );
        
        cascade->stage_classifier[i].count = cascade_src->stage_classifier[i].count;

        for( j = 0; j < cascade->stage_classifier[i].count; ++j )
        {
            cascade->stage_classifier[i].classifier[j].haar_feature = NULL;
        }

        for( j = 0; j < cascade->stage_classifier[i].count; ++j )
        {
            const CvHaarClassifier* classifier_src = 
                &cascade_src->stage_classifier[i].classifier[j];
            CvHaarClassifier* classifier = 
                &cascade->stage_classifier[i].classifier[j];

            classifier->count = classifier_src->count;
            CV_CALL( classifier->haar_feature = (CvHaarFeature*) cvAlloc( 
                classifier->count * ( sizeof( *classifier->haar_feature ) +
                                      sizeof( *classifier->threshold ) +
                                      sizeof( *classifier->left ) +
                                      sizeof( *classifier->right ) ) +
                (classifier->count + 1) * sizeof( *classifier->alpha ) ) );
            classifier->threshold = (float*) (classifier->haar_feature+classifier->count);
            classifier->left = (int*) (classifier->threshold + classifier->count);
            classifier->right = (int*) (classifier->left + classifier->count);
            classifier->alpha = (float*) (classifier->right + classifier->count);
            for( k = 0; k < classifier->count; ++k )
            {
                classifier->haar_feature[k] = classifier_src->haar_feature[k];
                classifier->threshold[k] = classifier_src->threshold[k];
                classifier->left[k] = classifier_src->left[k];
                classifier->right[k] = classifier_src->right[k];
                classifier->alpha[k] = classifier_src->alpha[k];
            }
            classifier->alpha[classifier->count] = 
                classifier_src->alpha[classifier->count];
        }
    }

    __END__;

    return cascade;
}


CvType haar_type( CV_TYPE_NAME_HAAR, icvIsHaarClassifier,
                  (CvReleaseFunc)cvReleaseHaarClassifierCascade,
                  icvReadHaarClassifier, icvWriteHaarClassifier,
                  icvCloneHaarClassifier );

/* End of file. */
