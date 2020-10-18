#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <list>

namespace JIT
{
	class Ptx
	{
		private:
			CUmodule cuModule;

			struct kernelHashTable
			{
				std::string ptx;
				std::string cu;
			};
			std::list<kernelHashTable> hashTable;

			void ptxParser(const char* ptxfilepath);
			void cuParser(const char* sourcefilepath);
		public:

			Ptx() { };
			~Ptx();
					
			void Load(const char* ptxfilepath, const char* sourcefilepath);
			CUmodule getModule();
			const char* getFunctionName(const std::string cuFunctionName);
	};
}