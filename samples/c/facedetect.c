#include "cv.h"
#include "highgui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#include <windows.h>

#ifdef _EiC
#define WIN32
#endif

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw( IplImage* image );

const char* cascade_name =
    "haarcascade_frontalface_alt.xml";
/*    "haarcascade_profileface.xml";*/

int wasInit;

int main( int argc, char** argv )
{
	// структура для получения видеопотока из камеры или AVI-файла
    CvCapture* capture = 0;

	// кадр изображения
    IplImage *frame, *frame_copy = 0;

	int optlen = strlen("--cascade=");
    const char* input_name;

    if( argc > 1 && strncmp( argv[1], "--cascade=", optlen ) == 0 )
    {
        cascade_name = argv[1] + optlen;
        input_name = argc > 2 ? argv[2] : 0;
    }
    else
    {
        cascade_name = "../../data/haarcascades/haarcascade_frontalface_alt.xml"; //было ..._alt2.xml
        input_name = argc > 1 ? argv[1] : 0;
    }

    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
    
    if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }

	wasInit = 1;
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    storage = cvCreateMemStorage(0);
    
    if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )
        capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );
    else
        capture = cvCaptureFromAVI( input_name ); 
//	capture = cvCaptureFromAVI( "E:\\__\\Warehouse.13.s02e06.rus.LostFilm.TV.avi" ); 

	// создаем окно для отображения
    cvNamedWindow( "result", 1 );

	// если пытались открыть видеопоток
    if( capture )
    {
        for(;;)
        {
			// получаем очередной фрейм и загружаем его в IplImage
            if( !cvGrabFrame( capture ))
                break;
            frame = cvRetrieveFrame( capture );

			// если фрейм не получен - ошибка
            if( !frame )
                break;

			// выделяем память под очередной фрейм
            if( !frame_copy )
                frame_copy = cvCreateImage( cvSize(frame->width,frame->height),
                                            IPL_DEPTH_8U, frame->nChannels );
            
			// ???
			if( frame->origin == IPL_ORIGIN_TL )
                cvCopy( frame, frame_copy, 0 );
            else
                cvFlip( frame, frame_copy, 0 );
            
			// осущетвляем поиск лица на изображении
            detect_and_draw( frame_copy );

			// ожидаем результат????
            if( cvWaitKey( 10 ) >= 0 )
                break;
        }

		// Release the images, and capture memory
        cvReleaseImage( &frame_copy );
        cvReleaseCapture( &capture );
    }
    else
    {
		// иначе загружаем изображение
        //const char* filename = input_name ? input_name : (char*)"lena.jpg";
		const char* filename = input_name ? input_name : (char*)"../../frame_0_df_2.jpg";
        IplImage* image = cvLoadImage( filename, 1 );

        if( image )
        {
            detect_and_draw( image, filename );
            cvWaitKey(0);
            cvReleaseImage( &image );
        }
        else
        {
            /* assume it is a text file containing the
               list of the image filenames to be processed - one per line */
            FILE* f = fopen( filename, "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    image = cvLoadImage( buf, 1 );
                    if( image )
                    {
                        detect_and_draw( image, filename );
                        cvWaitKey(0);
                        cvReleaseImage( &image );
                    }
                }
                fclose(f);
            }
        }

    }
    
	cvHaarDetectObjectsGPU_ClearMemory();
    cvDestroyWindow("result");
    //getch();
    return 0;
}

void detect_and_draw( IplImage* img, char *fileName )
{
	char fileNameFinal[50];
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };

    double scale = 1.1;

	CvSeq* faces;
	double t1, t2;
	int len = 0;

    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );	//????
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                         cvRound (img->height/scale)),
                     8, 1 );
    int i;

    cvCvtColor( img, gray, CV_BGR2GRAY );				//??
    cvResize( gray, small_img, CV_INTER_LINEAR );		//??
    cvEqualizeHist( small_img, small_img );				//??
    cvClearMemStorage( storage );

    if( cascade )
    {
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//								GPU COMPUTE
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        double frq = (double)cvGetTickFrequency();
//		int imageIndex;


/*
CVAPI(CvSeq*) cvHaarDetectObjectsGPU_Initialize( const CvArr* image,
                     CvHaarClassifierCascade* cascade,
                     CvMemStorage* storage,
					 double scale_factor CV_DEFAULT(1.1),
                     CvSize min_size CV_DEFAULT(cvSize(0,0)));
*/
		if (wasInit == 1)
		{
			cvHaarDetectObjectsGPU_Initialize( small_img, cascade, storage, scale, cvSize(32, 32) );
			wasInit = 0;
		}

//#define ImageCount 20
 //		for (imageIndex = 0; imageIndex < ImageCount; imageIndex++)
//		{
//			printf("\tNext image\n");
//			t1 = (double)cvGetTickCount();

/*
CVAPI(CvSeq*) cvHaarDetectObjectsGPU_Execute( const CvArr* image, CvMemStorage* storage, int min_neighbors CV_DEFAULT(3));
*/
			faces = cvHaarDetectObjectsGPU_Execute( small_img, storage, 4 );

//			t2 = (double)cvGetTickCount();
//			printf( "Image %d. Detection time = %g seconds\n", imageIndex + 1, (t2-t1)/(frq * 1000000) );

			for( i = 0; i < (faces ? faces->total : 0); i++ )
			{
				CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
				CvPoint center;
				int radius;
				center.x = cvRound((r->x + r->width*0.5)*scale);
				center.y = cvRound((r->y + r->height*0.5)*scale);
				radius = cvRound((r->width + r->height)*0.25*scale);
				cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
				printf("x = %d. y = %d. radius = %d\n", center.x, center.y, radius); 
			}

			//len = strlen(fileName) - 1;
			//while (len > 0 && fileName[len] != '.')
			//	len--;
			//strcpy_s(fileNameFinal, 50, fileName);
			//strcpy_s(&fileNameFinal[len], 6, ".GPU_");
			//sprintf(&fileNameFinal[len + 5], "%d", imageIndex);
			//strcpy_s(&fileNameFinal[strlen(fileNameFinal)], strlen(&fileName[len]) + 10, &fileName[len]);
			//cvSaveImage( fileNameFinal,  img);
			//cvClearMemStorage( storage );
//			system("PAUSE");
//		}
    }

	// отображение картинки в окне
	cvShowImage( "result", img );
	cvReleaseImage( &gray );			//??
	cvReleaseImage( &small_img );
///	system("PAUSE");
}
