#include "cudaDriverMode.h"
#include "cudaGenerator.h"
#include "Ptx.h"
#include <cxtypes.h>
#include <cvtypes.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "helper_cuda_drvapi.h"

#include <Windows.h>

using namespace JIT;

std::string cudaDriverMode::findVisualStudioPath()
{
	char path[512];
	GetEnvironmentVariable("VS100COMNTOOLS", path, 512);
	std::string _path(path);
	return _path + "..\\..\\VC\\bin";
}

std::string cudaDriverMode::findCUDAToolkitPath()
{
	char path[512];
	GetEnvironmentVariable("CUDA_PATH", path, 512);
	std::string _path(path);
	return _path + "\\include";
}

cudaDriverMode::cudaDriverMode(int cyclesOnGPU, int integralImageLength, int resLength, int resAllignedLength,
							   int sqsumLength, int sqsumAllignedLength, int plansLength, int elemLength,
							   void *_cascade_, int *profile, int length)
{
	sqsum_img_gpuD	  = new CUdeviceptr[cyclesOnGPU];
	integral_img_gpuD = new CUdeviceptr[cyclesOnGPU];
	res_gpuD		  = new CUdeviceptr[cyclesOnGPU];
	res_gpu_1cascadeD = new CUdeviceptr[cyclesOnGPU];
	elements_gpuD	  = new CUdeviceptr[cyclesOnGPU];
	plans_gpuD		  = new CUdeviceptr[cyclesOnGPU];
	_plans_gpuD		  = new CUdeviceptr[cyclesOnGPU];

	plans_cpu		= new unsigned short *[cyclesOnGPU];
	res_cpu			= new unsigned char *[cyclesOnGPU];
	varince_result	= new float *[cyclesOnGPU];
	_integral_img	= new unsigned int *[cyclesOnGPU];

	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&hDevice,  0));
	checkCudaErrors(cuCtxCreate(&hContext, 0, hDevice));

	stream = new CUstream[cyclesOnGPU];

	firstStageParam = new void*[cyclesOnGPU];
	filterParam = new void*[cyclesOnGPU];
	dataToQueueB1Param = new void*[cyclesOnGPU];
	nextStage1_Param = new void*[cyclesOnGPU];
	dataToQueueB = new void*[cyclesOnGPU];
	nextStage2_Param = new void*[cyclesOnGPU];
	initStreams = new int[cyclesOnGPU];

	cycles = cyclesOnGPU;

	for (int i = 0; i < cyclesOnGPU; i++)
	{
		checkCudaErrors(cuStreamCreate(&stream[i], 0));
		initStreams[i] = -1;

		checkCudaErrors(cuMemAlloc(&sqsum_img_gpuD[i],		sqsumAllignedLength ));
		checkCudaErrors(cuMemAlloc(&res_gpuD[i],			resAllignedLength ));
		checkCudaErrors(cuMemAlloc(&res_gpu_1cascadeD[i],	resAllignedLength ));
		checkCudaErrors(cuMemAlloc(&integral_img_gpuD[i],	integralImageLength ));
		checkCudaErrors(cuMemAlloc(&plans_gpuD[i],			plansLength ));
		checkCudaErrors(cuMemAlloc(&_plans_gpuD[i],		plansLength ));
		checkCudaErrors(cuMemAlloc(&elements_gpuD[i],		elemLength ));

		checkCudaErrors(cuMemAllocHost((void**)&_integral_img[i],  integralImageLength ));
		checkCudaErrors(cuMemAllocHost((void**)&varince_result[i], sqsumLength ));
		checkCudaErrors(cuMemAllocHost((void**)&res_cpu[i],		resLength ));
		checkCudaErrors(cuMemAllocHost((void**)&plans_cpu[i],	  plansLength ));

		memset( _integral_img[i],  0, integralImageLength );
		memset( varince_result[i], 0, sqsumLength );

		checkCudaErrors(cuMemsetD8(res_gpuD[i],			0, resAllignedLength));
		checkCudaErrors(cuMemsetD8(res_gpu_1cascadeD[i],	0, resAllignedLength));
		checkCudaErrors(cuMemsetD8(plans_gpuD[i],		0, plansLength));
		checkCudaErrors(cuMemsetD8(_plans_gpuD[i],		0, plansLength));
	}

	threads 	= dim3(16, 16);
	threads256  = dim3(256);

	printf(" > Generate kernel file\n");
	cascadeGenerator(_cascade_, profile, length);
	ptx = prepareToWork();

#ifdef _DEBUG
		fp = fopen("DEBUG.tmp", "w+");
#endif
}

cudaDriverMode::~cudaDriverMode(void)
{
	for (int i = 0; i < cycles; i++)
	{
		cuStreamDestroy(stream[i]);

		cuMemFree(sqsum_img_gpuD[i]);
		cuMemFree(res_gpuD[i]);
		cuMemFree(res_gpu_1cascadeD[i]);
		cuMemFree(integral_img_gpuD[i]);
		cuMemFree(plans_gpuD[i]);
		cuMemFree(_plans_gpuD[i]);
		cuMemFree(elements_gpuD[i]);

		cuMemFreeHost(_integral_img[i]);
		cuMemFreeHost(varince_result[i]);
		cuMemFreeHost(res_cpu[i]);
		cuMemFreeHost(plans_cpu[i]);
	}
	delete ptx;

	checkCudaErrors(cuCtxDestroy(hContext));

	delete firstStageParam;
	delete filterParam;
	delete dataToQueueB1Param;
	delete nextStage1_Param;
	delete dataToQueueB;
	delete nextStage2_Param;

	delete stream;
	delete sqsum_img_gpuD;
	delete res_gpuD;
	delete res_gpu_1cascadeD;
	delete integral_img_gpuD;
	delete plans_gpuD;
	delete _plans_gpuD;
	delete elements_gpuD;
	delete _integral_img;
	delete res_cpu;
	delete varince_result;
	delete plans_cpu;

	for (int i = 0; i < profileLength; i++)
	{
		for (int j = 0; j < profiles[i] - 1 && functionName[i][j] != NULL; j++)
			delete functionName[i][j];
		delete functionName[i];
	}
	delete functionName;

	fclose( fp );
}

void cudaDriverMode::setInputParameters(int integralImageLength, int sqsumAllignedLength, int blockDimX, int blockDimY, int blockDimX32, int blockDimY32,
										int integral_img_width, int integral_img_height, int integral_img_buffer_width, int windows_width32)
{
	this->integralImageLength = integralImageLength;
	this->sqsumAllignedLength = sqsumAllignedLength;
	blocks  	= dim3(blockDimX, blockDimY);
	blocks32 	= dim3(blockDimX32, blockDimY32);
	this->integral_img_width = integral_img_width;
	this->integral_img_height = integral_img_height;
	this->integral_img_buffer_width = integral_img_buffer_width;
	this->windows_width32 = windows_width32;
}

void cudaDriverMode::calcVarainceNormfactor(int pos, double *sqintegral_img, unsigned int *integral_img, int integral_img_height,
							 const int integral_img_width, const int integral_img_buffer_width, 
							 const int window_stepX, const int window_stepY,
							 const int windows_height, const int windows_width, const int windows_width32,
							 const int max_threads)
{
	const double tmp = 1.0/(18.0*18.0);
	unsigned int *_int_image = _integral_img[pos];
	float *_var_result = varince_result[pos];

#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(_int_image, integral_img)
#endif
	for (int i=0;i<integral_img_height;i++)
		for (int j=0;j<integral_img_width;j++)
			_int_image[integral_img_buffer_width*i + j] = integral_img[integral_img_width*i + j];

#ifdef _OPENMP
	#pragma omp parallel for num_threads(max_threads) shared(integral_img, sqintegral_img, _var_result)
#endif
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
			_var_result[windows_width32*(i-1)+(j-1)] = (float) variance_norm_factor;
		}
	checkCudaErrors(cuMemcpyHtoDAsync(integral_img_gpuD[pos],	_integral_img[pos],		integralImageLength, stream[pos] ));
	checkCudaErrors(cuMemcpyHtoDAsync(sqsum_img_gpuD[pos],	varince_result[pos],	sqsumAllignedLength, stream[pos] ));
}

void* cudaDriverMode::prepareToWork()
{
	printf(" > Compile kernel file\n");

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	char computeCapability[3];
	computeCapability[0] = (char)(deviceProp.major + '0');
	computeCapability[1] = (char)(deviceProp.minor + '0');
	computeCapability[2] = '\0';

	std::string compilePath("nvcc.exe -arch sm_");
	compilePath += computeCapability;
	compilePath += " -maxrregcount=32 -m32 -ptx -ccbin \"";
	compilePath += findVisualStudioPath();
	compilePath += "\" -I\"";
	compilePath += findCUDAToolkitPath();
	compilePath += "\" -o kernelFiles.ptx \"kernel.cu\" ";

	std::cout << compilePath << std::endl;
	
	system(compilePath.c_str());

	Ptx* ptx = new Ptx();
	printf(" > Load binary file\n");
	ptx->Load("kernelFiles.ptx", "kernel.cu");

	cuModule = ptx->getModule();

	printf("Compilation success\n");

	return ptx;
}

void *cudaDriverMode::getHaarNextStageParams( unsigned int *plansPointer, int pos, int index, int start )
{
	Ptx *_ptx = (Ptx *)ptx;

	CUfunction vecAdd_kernel;
	unsigned tmp = 0;
	checkCudaErrors( cuModuleGetFunction(&vecAdd_kernel, cuModule, _ptx->getFunctionName(functionName[pos][start])) );
	void *args[] = { &integral_img_gpuD[pos],
					 &sqsum_img_gpuD[pos],

					 &res_gpuD[pos],
					 &integral_img_width,
					 &integral_img_height,
					 &integral_img_buffer_width,
					 &windows_width32,
					 &tmp,
					 &plansPointer[pos],
					 &elements_gpuD[pos] };

	checkCudaErrors ( cuLaunchKernel(vecAdd_kernel,  blocks32.x, blocks32.y, blocks32.z,
                               threads.x, threads.y, threads.z,
                               0,
                               stream[pos], args, NULL) );
	return NULL;
}
void* cudaDriverMode::getFilterParams(int pos, int index, int blockDimY256_32, int blockDimX32)
{
	Ptx *_ptx = (Ptx *)ptx;

	CUfunction vecAdd_kernel;
	unsigned tmp = 0;
	checkCudaErrors( cuModuleGetFunction(&vecAdd_kernel, cuModule, _ptx->getFunctionName("filter")) );
	void *args[] = { &res_gpu_1cascadeD[pos],
					 &res_gpuD[pos],
					 &windows_width32 };

	checkCudaErrors ( cuLaunchKernel(vecAdd_kernel,  blockDimY256_32, blockDimX32, 1,
                               threads256.x, threads256.y, threads256.z,
                               0,
                               stream[pos], args, NULL) );
	return NULL;
}

void* cudaDriverMode::getDataToQueueB1_32Params(int pos, int index)
{
	Ptx *_ptx = (Ptx *)ptx;

	CUfunction vecAdd_kernel;
	unsigned tmp = 0;
	checkCudaErrors( cuModuleGetFunction(&vecAdd_kernel, cuModule, _ptx->getFunctionName("DataToQueueB1_32")) );
	void *args[] = { &res_gpuD[pos],
					 &plans_gpuD[pos],
					 &elements_gpuD[pos] };

	checkCudaErrors ( cuLaunchKernel(vecAdd_kernel,  blocks32.x, blocks32.y, blocks32.z,
                               threads256.x, threads256.y, threads256.z,
                               0,
                               stream[pos], args, NULL) );
	return NULL;
}

void* cudaDriverMode::getDataToQueueBParams(int pos, int index)
{
	Ptx *_ptx = (Ptx *)ptx;

	CUfunction vecAdd_kernel;
	unsigned tmp = 0;
	checkCudaErrors( cuModuleGetFunction(&vecAdd_kernel, cuModule, _ptx->getFunctionName("DataToQueueB")) );
	void *args[] = { &res_gpuD[pos],
					 &plans_gpuD[pos],
					 &_plans_gpuD[pos],
					 &elements_gpuD[pos] };

	checkCudaErrors ( cuLaunchKernel(vecAdd_kernel,  blocks32.x, blocks32.y, blocks32.z,
                               threads256.x, threads256.y, threads256.z,
                               0,
                               stream[pos], args, NULL) );
	return NULL;
}
void *cudaDriverMode::getHaarFirstStageParams(int pos, int index)
{
	Ptx *_ptx = (Ptx *)ptx;

	CUfunction vecAdd_kernel;
	unsigned tmp = 0;
	const char *funcName = _ptx->getFunctionName("haar_first_stage_1");
	printf("haar_first_stage_1. PTX name: %s\n", funcName);
	checkCudaErrors( cuModuleGetFunction(&vecAdd_kernel, cuModule, funcName) );
	void *args[] = { &integral_img_gpuD[pos],
					 &sqsum_img_gpuD[pos],
					 &res_gpu_1cascadeD[pos],
					 &integral_img_width,
					 &integral_img_height,
					 &integral_img_buffer_width,
					 &windows_width32,
					 &tmp };

	checkCudaErrors ( cuLaunchKernel(vecAdd_kernel,  blocks.x, blocks.y, blocks.z,
                               threads.x, threads.y, threads.z,
                               0,
                               stream[pos], args, NULL) );

	return NULL;
}

void cudaDriverMode::execute(int pos, int windows_width, int windows_height, int blockDimY256_32,
							 int resLength, int planLength, int blockDimX32, int blockDimY32,
							 unsigned char **_res_cpu, unsigned short **_plans_cpu)
{
	integral_img_width--;
	integral_img_height--;
	windows_width--;
	windows_height--;
	CUdeviceptr plans_gpu_tmp;

	initStreams[pos] = pos;
	Ptx *_ptx = (Ptx *)ptx;

	getHaarFirstStageParams(pos, 0);
	getFilterParams(pos, 1, blockDimY256_32, blockDimX32);
	getDataToQueueB1_32Params(pos, 2);
	getHaarNextStageParams(plans_gpuD, pos, 3, 0);

	int start = 1;
	int _pos_ = 4;
	while (functionName[pos][start] != NULL)
	{
		getDataToQueueBParams(pos, _pos_);
		_pos_++;
		getHaarNextStageParams(_plans_gpuD, pos, _pos_,start);
		_pos_++;
		plans_gpu_tmp	 = plans_gpuD[pos];
		plans_gpuD[pos]	 = _plans_gpuD[pos];
		_plans_gpuD[pos] = plans_gpu_tmp;
		start++;
	}

	checkCudaErrors( cuMemcpyDtoHAsync(res_cpu[pos], res_gpuD[pos], resLength, stream[pos] ));
	checkCudaErrors( cuMemcpyDtoHAsync(plans_cpu[pos], _plans_gpuD[pos], planLength, stream[pos] ));

	*_res_cpu = res_cpu[pos];
	*_plans_cpu = plans_cpu[pos];

	printf("Execution start success\n");
}
int cudaDriverMode::synchronize(int streamsOnGPU)
{
	// Бесконечный цикл, до тех пор, пока не найдется stream, который завершился
	while (true)
	{
		for (int i = 0; i < streamsOnGPU; i++)
		{
/*
	4.30.2.3 cuStreamQuery (CUstream hStream)
	Returns CUDA_SUCCESS if all operations in the stream speciﬁed by hStream have completed, or CUDA_ERROR_NOT_READY if not.
*/
			if ( initStreams[i] != -1 && cuStreamQuery(stream[initStreams[i]]) == CUDA_SUCCESS )
			{
				int pos = initStreams[i];
				initStreams[i] = initStreams[streamsOnGPU - 1];
				initStreams[streamsOnGPU - 1] = -1;
				return pos;
			}
		}
	}
}

void cudaDriverMode::cascadeGenerator(void *_cascade_, int *profile, int length/*, char ***functionName*/)
{
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade *)_cascade_;
	// булевые переменные используются для определения: были ли сгенерированы ядра
	// Если элемент массива = true, то данное ядро было сгенерировано

	// максимальное число каскадов, которое может быть в одном ядре
	#define genCount 10
	bool **wasGenerated = new bool*[genCount];

	functionName = new char **[length];
	profiles = profile;
	profileLength = length;

	for (int j = 0; j < genCount; j++)
	{
		wasGenerated[j] = new bool[profile[0]];
		for (int i = 0; i < profile[0]; i++)
			wasGenerated[j][i] = false;
	}
/*
void GenerateCascade(CvHaarClassifierCascade* cascade, int cascadeStart, int cascadeStop, int cascadePerKernel, int threadsPerPixel)
	cascadeStart		- начальный каскад классификатора (начинается с 0)
	cascadeStop			- (финальный каскад классификатора, который требуется учитывать, + 1). Минимум = 1
	cascadePerKernel	- число каскадов, которые будут обрабатываться одним ядром
	threadsPerPixel		- число тредов на один элемент изображения
*/
	cudaGenerator generator;

	FILE *kernelFile = fopen("kernel.cu","wb");
	generator.generateHeader(kernelFile);

	// перебираем все коэффициенты масштабирования для GPU
	for (int i = 0; i < length; i++)
	{
		functionName[i] = new char*[profile[i] - 1];
		// генерация ядра haar_first_stage
		generator.GenerateCascade(cascade, 0, 2, 0, 1, wasGenerated, kernelFile, NULL, &functionName[i], 0);

//										Настройка параметров генератора ядер
//----------------------------------------------------------------------------------------------------------------------------------------------
		int position = 0;
		// это натройка на наиболее оптимальную комбинацию из 1- или 2- каскадов на 1 ядро, обрабатываемых на Core2Duo E4400 + GeForce 9800 GT Green
		if (profile[i] >= 4)
		{
			// со 2 по 4 каскад обрабатываем в режиме "1 каскад = 1 ядро"
			position = generator.GenerateCascade(cascade, 2,  4,  1,  2, wasGenerated, kernelFile, NULL, &functionName[i], position);

// ТЕСТОВЫЙ ШАБЛОН. ТЕСТ НА ОБРАБОТКУ ПО 3 КАСКАДА
			// каскады с 4 и до конца обрабатываются по 3 штуки на ядро. Дополнительно осуществляется проверка
			//			на доработку последних 2 каскадов
			if ( ( profile[i] - 4 ) % 3 == 0 )
			{
				// число каскадов делится на 3 без остатка
				//		значит генерируем оставшиеся ядра по 3 каскада на 1 ядро
				position = generator.GenerateCascade(cascade,			  4, profile[i], 3, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
			}

			if ( ( profile[i] - 4 ) % 3 == 1 )
			{
				// последний элемент дообрабатывается в режиме 1 каскад = 1 ядро
				position = generator.GenerateCascade(cascade,			   4, profile[i] - 1, 3, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
				position = generator.GenerateCascade(cascade, profile[i] - 1,     profile[i], 1, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
			}

			if ( ( profile[i] - 4 ) % 3 == 2 )
			{
				// последний элемент дообрабатывается в режиме 2 каскада = 1 ядро
				position = generator.GenerateCascade(cascade,			   4, profile[i] - 2, 3, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
				position = generator.GenerateCascade(cascade, profile[i] - 2,     profile[i], 2, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
			}

// ТЕСТОВЫЙ ШАБЛОН. РАБОЧАЯ ВЕРСИЯ
//			// если число каскадов нечетное, то до последнего каскада обрабатываем
//			//		в режиме "2 каскада = 1 ядро".
//			//		последний каскад = 1 ядро
//			// если число каскадов четное, то режим "2 каскада = 1 ядро"
//			
//			if (profile[i] & 1 == 1)
//			{
//				// нечетное кол-во
//				position = generator.GenerateCascade(cascade,			  4, profile[i] - 1, 2, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
//				position = generator.GenerateCascade(cascade, profile[i] - 1, profile[i],	 1, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
//			}
//			else
//				// четное кол-во
//				position = generator.GenerateCascade(cascade, 4, profile[i], 2, 4, wasGenerated, kernelFile, NULL, &functionName[i], position);
		}
		else
			position = generator.GenerateCascade(cascade, 2,  profile[i],  1,  2, wasGenerated, kernelFile, NULL, &functionName[i], 0);
//----------------------------------------------------------------------------------------------------------------------------------------------

		functionName[i][position] = NULL;
	}
	fclose(kernelFile);

	for (int i = 0; i < genCount; i++)
		delete[] wasGenerated[i];
	delete[] wasGenerated;
}
