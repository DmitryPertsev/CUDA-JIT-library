#ifndef __PROFILES_H__
#define __PROFILES_H__
	
enum  ProfileName { I7_920_GFX285 = 0, Q9400_9800GT, Q9400_9800GT_JIT, Core2Duo_E4400_9800GT_Green };


class ComputeProfile
{
	public:
		int length;
		int* profile;
		ComputeProfile (int length, int* profile)
		{
			this->length = length;
			this->profile = profile;
		}
		
};

ComputeProfile* GetProfile(ProfileName name);


#endif