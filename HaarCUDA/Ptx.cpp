#include "Ptx.h"

#include <string>
#include <iostream>
#include <fstream>
#include <regex>

#include "helper_cuda_drvapi.h"

namespace JIT
{
	void Ptx::ptxParser(const char* ptxFilePath)
	{
		std::ifstream _file(ptxFilePath, std::ios::binary);
		std::string buffer(
			(std::istreambuf_iterator<char>(_file)),
			std::istreambuf_iterator<char>()
		);
		std::tr1::regex pattern(".entry( )+[A-Za-z0-9_]+");
		const std::tr1::sregex_token_iterator end;

		for ( std::tr1::sregex_token_iterator i( buffer.begin(), buffer.end(), pattern );
			  i != end; i++)
		{
			std::string wasFound = (std::string)(*i);
			kernelHashTable hash;
			hash.ptx = wasFound.substr(wasFound.find_last_of(' ') + 1);
			hashTable.push_back( hash );
		}
	}

	void Ptx::cuParser(const char* sourceFilePath)
	{
		std::ifstream _file(sourceFilePath, std::ios::binary);
		std::string buffer(
			(std::istreambuf_iterator<char>(_file)),
			std::istreambuf_iterator<char>()
		);
		std::tr1::regex pattern("__global__ +(void|static( )+void)( )+(\\w)+");
		const std::tr1::sregex_token_iterator end;

		std::list<kernelHashTable>::iterator it = hashTable.begin();
		unsigned int wasAnalyzed = 0;

		for ( std::tr1::sregex_token_iterator i( buffer.begin(), buffer.end(), pattern );
			  i != end; i++)
		{
			std::string wasFound = (std::string)(*i);
			std::string functionName = wasFound.substr(wasFound.find_last_of(" ") + 1);
			if ( it != hashTable.end() )
			{
				it->cu = functionName;
				it++;
			}
			else
				std::cerr<<"PARSER ERROR. Functions count mismatch"<<std::endl;
		}
	}

	Ptx::~Ptx()
	{
		hashTable.clear();
		checkCudaErrors( cuModuleUnload(cuModule) );
	}

	void Ptx::Load(const char* ptxfilepath, const char* sourcefilepath)
	{
		const unsigned jitNumOptions = 3;
		const unsigned jitLogBufferSize = 1024;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];
		char jitLogBuffer[jitLogBufferSize];

		{
			// set up size of compilation log buffer
			jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
			jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

			// set up pointer to the compilation log buffer
			jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
			jitOptVals[1] = jitLogBuffer;

			// set up pointer to set the Maximum # of registers for a particular kernel
			jitOptions[2] = CU_JIT_MAX_REGISTERS;
			int jitRegCount = 32;
			jitOptVals[2] = (void *)(size_t)jitRegCount;
		}

		std::ifstream file (ptxfilepath, std::ifstream::in);
		std::string content( (std::istreambuf_iterator<char>(file) ),
							 (std::istreambuf_iterator<char>()    ) );

		checkCudaErrors( cuModuleLoadDataEx(&cuModule, content.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals) );

		std::cout << "> PTX JIT log:" << std::endl << jitLogBuffer << std::endl;
		delete jitOptions;
		delete jitOptVals;

		ptxParser(ptxfilepath);
		 cuParser(sourcefilepath);
	}

	CUmodule Ptx::getModule()
	{
		return cuModule;
	}

	const char* Ptx::getFunctionName(const std::string cuFunctionName)
	{
		std::list<kernelHashTable>::iterator it = hashTable.begin();
		for ( ; it != hashTable.end(); it++)
			if ( it->cu == cuFunctionName )
				return it->ptx.c_str();
		return NULL;
	}
}