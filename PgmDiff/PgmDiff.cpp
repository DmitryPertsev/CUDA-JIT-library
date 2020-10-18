// PgmDiff.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "stdlib.h"
#include "memory.h"
#include "svldpgm.h"


int main(int argc, char* argv[])
{
	if (argc<3)
	    return 0;
	FILE *fp = fopen(argv[1], "rb");
	if (fp == 0)
	{   printf ("File not found %s\n",argv[1]);
	    return 0;}
	    
	int width1, height1;
	unsigned char *data1;
	load_pgm(fp,&width1, &height1, &data1);
	fclose (fp);
	
	fp = fopen(argv[2], "rb");
	if (fp == 0)
	{   printf ("File not found %s\n",argv[2]);
	    return 0;}
	    
	int width2, height2;
	unsigned char *data2;
	load_pgm(fp,&width2, &height2, &data2);
	fclose (fp);
	
	if (width1!=width2)
	{
	    printf ("width1(&d) != width2(%d)",width1,width2 );
	    return 0;}
	    
	if (height1!=height2)
	{
	    printf ("height1(&d) != height2(%d)",height1,height2 );
	    return 0;}
	
	unsigned char *res = (unsigned char *)malloc(width1*height1*sizeof(unsigned char));
	memset(res, 128, width1*height1*sizeof(unsigned char));
	int flag = 0;
	for (int i=0;i<height1;i++)
	    for (int j=0;j<width1;j++)
	    {
	        if (data1[width1*i+j]!=data2[width1*i+j])
	        {
	            res[width1*i+j] = data2[width1*i+j];
	            flag ++;
	            }
	    }
	if (flag)
	    printf("There are %d differenses\n", flag);
	fp = fopen(argv[3], "wb");
	if (fp == 0)
	{   printf ("File not found %s\n",argv[3]);
	    return 0;}
	save_pgm(fp,width2, height2, res);
	fclose (fp)   ;
	
	return 0;
}

/*etalon1.6.pgm dump_factor1.6.pgm diff.pgm*/
