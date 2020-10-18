#include <_cv.h>
#include "Profiles.h"

// описание числа каскадов, которые будут обрабатываться на GPU, в зависимости от значения factor
int I7_920_GFX285p[16]		= {12, 11, 11, 11, 10,  9, 9, 8, 8, 7, 7, 6, 5, 4, 4, 4};
int Q9400_9800GTp[16]		= {12, 11, 11, 11, 10,  9, 9, 8, 8, 7, 7, 6, 5, 4, 4, 4};
//int Q9400_9800GT_JITp[19]	= {13, 12, 12, 12, 11, 10, 9, 8, 8, 8, 8, 6, 5, 5, 5, 5, 4, 4, 3};
int Q9400_9800GT_JITp[19]	= {13, 12, 12, 12, 11, 10, 9, 8, 8, 8, 8, 6, 5, 5, 5, 5, 4, 4, 3};

int E4400_9800GT_JITp[19]	= {13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};

ComputeProfile* GetProfile(ProfileName name)
{
	switch (name)
	{
		case ProfileName::I7_920_GFX285:
			return new ComputeProfile(16,I7_920_GFX285p);
			break;
		case ProfileName::Q9400_9800GT:
			return new ComputeProfile(16,Q9400_9800GTp);
			break;
		case ProfileName::Q9400_9800GT_JIT:
			return new ComputeProfile(19,Q9400_9800GT_JITp);
			break;
		case ProfileName::Core2Duo_E4400_9800GT_Green:
			return new ComputeProfile(19,E4400_9800GT_JITp);
			break;
			
		default:
				break;
	}
}
