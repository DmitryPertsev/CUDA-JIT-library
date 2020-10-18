/* AUTHOR: Dmitry A. Bushenko
 * CREATED: 05.09.2006
 * 
 * MODULE DESCRIPTION
 * Contains functions for reading and writing pgm files.
 * Uses libnetpbm-dev package. Must be linked with libnetpbm.
 */
  
//#include <pgm.h>
//#include "svldpgm.h"
#include "malloc.h"
#include "stdio.h"
#include "string.h"
#include <stdlib.h>

 


/****************************************************
 * Description:
 *  Loads the pgm file into a plain array
 * Parameters:
 *  fp -- opened file stream
 *  width -- here will be stored an image width
 *  height -- here will be stored an image height
 *  data -- here will be stored the image data
 * Returns:
 *  0 -- if success, -1 -- if error
 */

/*
P5
# CREATOR: GIMP PNM Filter Version 1.1
300 300
255*/

void skip_white_space(FILE* fp)
{
    char ch;
    fread(&ch,1,1,fp);
    while (ch==' ' || ch == '\r' || ch == '\n' || ch == '\t')
        fread(&ch,1,1,fp);
    long t = ftell(fp);
    fseek(fp,-1,SEEK_CUR);
    t = ftell(fp);
}
int isComment (FILE* fp)
{
    char ch;
    fread(&ch,1,1,fp);
    if ('#' == ch)
        return 1;

    fseek(fp,-1,SEEK_CUR);
    return 0;
}
void pgm_readpgminit (FILE* fp, int* width, int* height, int *maxval, int *format)
{
    int state = 0;
    int comment = 1;
    char buffer[255];
    char header[16];
    while (1)
    {
        skip_white_space(fp);
        if (isComment(fp))
        {
            char ch;
            fread(&ch,1,1,fp);
            while (ch!='\n') 
                fread(&ch,1,1,fp);
        }
        else if (state == 0)
        {
            char *p = buffer;
            fread(p,1,1,fp);
            while (*p!='\n') 
            {
                p++;
                fread(p,1,1,fp);
            }
            *p = 0;
            strcpy_s(header,16, buffer);       
            state = 1; 
        }
        else if (state ==1)
        {
            char *p = buffer;
            fread(p,1,1,fp);
            while (*p!=' ') 
            {
                p++;
                fread(p,1,1,fp);
            }
            *p = 0;
            *width = atoi(buffer);

            skip_white_space(fp);

            p = buffer;
            fread(p,1,1,fp);
            while (*p!='\n') 
            {
                p++;
                fread(p,1,1,fp);
            }
            *p = 0;
            *height = atoi(buffer);
            state = 2;
        }
        else if (state ==2)
        {
            char *p = buffer;
            fread(p,1,1,fp);
            while (*p!='\n') 
            {
                p++;
                fread(p,1,1,fp);
            }
            *p = 0;
            *maxval = atoi(buffer);
            skip_white_space(fp);
            break;
        }
    }
}

int load_pgm(FILE* fp,int* width, int* height, unsigned char** data)
{
	int format;
	int maxval;
	unsigned long int size;

	pgm_readpgminit(fp,width,height,&maxval,&format);
	size = (*width) * (*height);
	
	*data = (unsigned char*) malloc( size );
	int res = fread(*data,size,1,fp);
	
	return -(1 - res);
}

/*******************************************
 * Description:
 *  Saves data to pgm file
 * Parameters:
 *  fp -- opened file stream
 *  width -- image width
 *  height -- image height
 *  data -- image data
 * Returns:
 *  0 -- if success, -1 -- if error
 */
int save_pgm(FILE* fp,int width, int height, unsigned char* data)
{
	fprintf(fp,"P5\n%d %d\n255\n",width,height);
	int res = fwrite(data,width*height,1,fp);
	
	fflush(fp);
	
	return -(1 - res);
}

int save_pgmt(FILE* fp,int width, int height, unsigned char* data)
{
    fprintf(fp,"P5\n%d %d\n255\n",width,height);
	for (int i =0; i< height;i++)
	{
	    for (int j=0; j<width;j++)
	        fprintf(fp,"%4d ",data[width*i+j]);
	    fprintf(fp,"\n");
	}
	
	fflush(fp);
	
	return 1;
}


int save_pgm4(FILE* fp,int width, int height, unsigned int* data)
{
	fprintf(fp,"P40\n%d %d\n4228250625\n",width,height);
	int res = fwrite(data,width*height,4,fp);
	
	fflush(fp);
	
	return -(1 - res);
}

int save_pgm4t(FILE* fp,int width, int height, unsigned int* data)
{
	fprintf(fp,"P40\n%d %d\n4228250625\n",width,height);
	for (int i=0;i<height;i++)
	{
	    for (int j = 0; j<width;j++)
	        fprintf(fp,"%4d ", data[i*width+j]);
	    fprintf(fp,"\n" );
    }
	        
	int res = height*width;
	fflush(fp);
	
	return -(1 - res);
}

int save_pgm8f(FILE* fp,int width, int height, double* data)
{
	fprintf(fp,"P400\n%d %d\n4228250625\n",width,height);
	int res = fwrite(data,width*height,8,fp);
	
	fflush(fp);
	
	return -(1 - res);
}

int load_pgm4(FILE* fp,int* width, int* height, unsigned int** data)
{
	int format;
	int maxval;
	unsigned long int size;

	pgm_readpgminit(fp,width,height,&maxval,&format);
	size = (*width) * (*height);
	
	*data = (unsigned int*) malloc( size*sizeof(unsigned int));
	int res = fread(*data,size,4,fp);
	return -(1 - res);
}

int load_pgm8f(FILE* fp,int* width, int* height, double** data)
{
	int format;
	int maxval;
	unsigned long int size;

	pgm_readpgminit(fp,width,height,&maxval,&format);
	size = (*width) * (*height);
	
	*data = (double*) malloc( size*sizeof(double));
	int res = fread(*data,size,8,fp);
	return -(1 - res);
}