#ifndef _GPU_FILTER_KERNEL_CU__
#define _GPU_FILTER_KERNEL_CU__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syscall.h>
#include "../../buffer/GpuStreamEventBuffer.h"
#include "../../buffer/GpuIntBuffer.h"
#include "../../domain/GpuMetaEvent.h"
#include "../../main/GpuProcessor.h"
#include "../../domain/GpuProcessorContext.h"
#include "../../util/GpuCudaHelper.h"
#include "../../util/GpuUtils.h"
#include "../../filter/GpuFilterProcessor.h"
#include "../../domain/GpuKernelDataTypes.h"
#include "../../filter/GpuFilterKernelCore.h"
#include "../../filter/GpuFilterKernel.h"
#include "../../filter/CpuFilterKernel.h"
#include "../../util/GpuCudaCommon.h"

#include <cub/cub.cuh>

namespace SiddhiGpu
{


// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds
#define THREADS_PER_BLOCK 128
#define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS 8


__global__
void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
ProcessEventsFilterKernelStandalone(
		KernelParametersFilterStandalone  * _pParameters,
		int                                 _iEventCount        // Num events in this batch
)
{
	if(threadIdx.x >= _pParameters->i_EventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iEventCount / _pParameters->i_EventsPerBlock) && // last thread block
			(threadIdx.x >= _iEventCount % _pParameters->i_EventsPerBlock))
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _pParameters->i_EventsPerBlock) + threadIdx.x;

	char * pEvent = _pParameters->p_InByteBuffer + (_pParameters->i_SizeOfEvent * iEventIdx);

	FilterEvalParameters mEvalParameters;
	mEvalParameters.p_Meta = _pParameters->p_MetaEvent;
	mEvalParameters.p_Filter = _pParameters->ap_Filter;
	mEvalParameters.p_Event = pEvent;
	mEvalParameters.i_CurrentIndex = 0;

	bool bResult = Evaluate(mEvalParameters);
	printf(">>>>>>>>>> bResult - %d\n", bResult);
	printf(">>>>>>>>>> iEventIdx - %d\n", iEventIdx);
	printf(">>>>>>>>>> Event - %d\n", mEvalParameters.p_Event);

	if(bResult)
	{
		_pParameters->p_ResultBuffer[iEventIdx] = iEventIdx + 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		_pParameters->p_ResultBuffer[iEventIdx] = -1 * (iEventIdx + 1);
	}
}

__global__
void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
ProcessEventsFilterKernelFirstV2(
		KernelParametersFilterStandalone  * _pParameters,
		int                                 _iEventCount        // Num events in this batch
)
{
	if(threadIdx.x >= _pParameters->i_EventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iEventCount / _pParameters->i_EventsPerBlock) && // last thread block
			(threadIdx.x >= _iEventCount % _pParameters->i_EventsPerBlock))
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _pParameters->i_EventsPerBlock) + threadIdx.x;

	char * pInEvent = _pParameters->p_InByteBuffer + (_pParameters->i_SizeOfEvent * iEventIdx);

	FilterEvalParameters mEvalParameters;
	mEvalParameters.p_Filter = _pParameters->ap_Filter;
	mEvalParameters.p_Meta = _pParameters->p_MetaEvent;
	mEvalParameters.p_Event = pInEvent;
	mEvalParameters.i_CurrentIndex = 0;

	bool bMatched = Evaluate(mEvalParameters);

	if(bMatched)
	{
		_pParameters->p_ResultBuffer[iEventIdx] = 1;
	}
	else // ~ possible way to avoid cudaMemset from host
	{
		_pParameters->p_ResultBuffer[iEventIdx] = 0;
	}
}

__global__
void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
ResultSorter(
		char               * _pInByteBuffer,      // Input ByteBuffer from java side
		int                * _pMatchedIndexBuffer,// Matched event index buffer
		int                * _pPrefixSumBuffer,   // prefix sum buffer
		int                  _iEventCount,        // Num events in original batch
		int                  _iSizeOfEvent,       // Size of an event
		int                  _iEventsPerBlock,    // number of events allocated per block
		char               * _pOutputEventBuffer  // Matched events final buffer
)
{
	if(threadIdx.x >= _iEventsPerBlock || threadIdx.y > 0 || blockIdx.y > 0)
		return;

	if((blockIdx.x == _iEventCount / _iEventsPerBlock) && // last thread block
			(threadIdx.x >= _iEventCount % _iEventsPerBlock))
	{
		return;
	}

	// get assigned event
	int iEventIdx = (blockIdx.x * _iEventsPerBlock) + threadIdx.x;

	if(_pMatchedIndexBuffer[iEventIdx] == 0)
	{
		return;
	}

	char * pInEventBuffer = _pInByteBuffer + (_iSizeOfEvent * iEventIdx);
	char * pOutEventBuffer = _pOutputEventBuffer + (_iSizeOfEvent * (_pPrefixSumBuffer[iEventIdx]));

	memcpy(pOutEventBuffer, pInEventBuffer, _iSizeOfEvent);

}

// ============================================================================================================

GpuFilterKernelStandalone::GpuFilterKernelStandalone(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_DeviceFilter(NULL),
	p_InputEventBuffer(NULL),
	p_ResultEventBuffer(NULL),
	p_DeviceParameters(NULL),
	b_DeviceSet(false)
{

}

GpuFilterKernelStandalone::~GpuFilterKernelStandalone()
{
	fprintf(fp_Log, "[GpuFilterKernelStandalone] destroy\n");
	fprintf(fp_Log, "[GpuFilterKernelStandalone] %d", p_DeviceFilter);
	fflush(fp_Log);

//	CUDA_CHECK_RETURN(cudaFree(p_DeviceFilter));
	p_DeviceFilter = NULL;

//	CUDA_CHECK_RETURN(cudaFree(p_DeviceParameters));
	p_DeviceParameters = NULL;

	if(p_DeviceOutputAttributeMapping)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceOutputAttributeMapping));
		p_DeviceOutputAttributeMapping = NULL;
	}

//	sdkDeleteTimer(&p_StopWatch);
//	p_StopWatch = NULL;
}

bool GpuFilterKernelStandalone::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	//TODO: remove
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);

	fprintf(fp_Log, "[GpuFilterKernelStandalone] Initialize : StreamIndex=%d\n", _iStreamIndex);
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuFilterKernelStandalone] InputEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*)p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	GpuMetaEvent * pFilterResultMetaEvent = new GpuMetaEvent(_pMetaEvent->i_StreamIndex, 1, sizeof(int));
	pFilterResultMetaEvent->SetAttribute(0, DataType::Int, sizeof(int), 0);

	p_ResultEventBuffer = new GpuIntBuffer("FilterResultEventBuffer", p_Context->GetDeviceId(), pFilterResultMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);
	fprintf(fp_Log, "[GpuFilterKernelStandalone] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();
    delete pFilterResultMetaEvent;


	fprintf(fp_Log, "[GpuFilterKernelStandalone] Copying filter to device \n");
	fflush(fp_Log);

	GpuFilterProcessor * pFilter = (GpuFilterProcessor*)p_Processor;

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &p_DeviceFilter,
			sizeof(GpuKernelFilter)));

	GpuKernelFilter * apHostFilters = (GpuKernelFilter *) malloc(sizeof(GpuKernelFilter));


	apHostFilters->i_NodeCount = pFilter->i_NodeCount;
	apHostFilters->ap_ExecutorNodes = NULL;

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &apHostFilters->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount));

	CUDA_CHECK_RETURN(cudaMemcpy(
			apHostFilters->ap_ExecutorNodes,
			pFilter->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount,
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_DeviceFilter,
			apHostFilters,
			sizeof(GpuKernelFilter),
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	free(apHostFilters);
	apHostFilters = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceParameters, sizeof(KernelParametersFilterStandalone)));
	KernelParametersFilterStandalone * pHostParameters = (KernelParametersFilterStandalone*) malloc(sizeof(KernelParametersFilterStandalone));

	pHostParameters->p_InByteBuffer = p_InputEventBuffer->GetDeviceEventBuffer();
	pHostParameters->ap_Filter = p_DeviceFilter;
	pHostParameters->p_MetaEvent = p_InputEventBuffer->GetDeviceMetaEvent();
	pHostParameters->p_ResultBuffer = p_ResultEventBuffer->GetDeviceEventBuffer();
	pHostParameters->i_SizeOfEvent = p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes;
	pHostParameters->i_EventsPerBlock = i_ThreadBlockSize;

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_DeviceParameters,
			pHostParameters,
			sizeof(KernelParametersFilterStandalone),
			cudaMemcpyHostToDevice));

	free(pHostParameters);
	pHostParameters = NULL;



	// copy Output mappings
	if(p_HostOutputAttributeMapping)
	{
		printf("ATTRIBUTE MAPPING \n");
		fprintf(fp_Log, "[GpuFilterKernelStandalone] Copying AttributeMappings to device \n");
		fflush(fp_Log);

		fprintf(fp_Log, "[GpuFilterKernelStandalone] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
		for(int c=0; c<p_HostOutputAttributeMapping->i_MappingCount; ++c)
		{
			fprintf(fp_Log, "[GpuFilterKernelStandalone] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::STREAM_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::ATTRIBUTE_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].to);

		}

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &p_DeviceOutputAttributeMapping,
				sizeof(AttributeMappings)));

		AttributeMappings * pHostMappings = (AttributeMappings*) malloc(sizeof(AttributeMappings));
		pHostMappings->i_MappingCount = p_HostOutputAttributeMapping->i_MappingCount;
		pHostMappings->p_Mappings = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &pHostMappings->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				pHostMappings->p_Mappings,
				p_HostOutputAttributeMapping->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount,
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceOutputAttributeMapping,
				pHostMappings,
				sizeof(AttributeMappings),
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		free(pHostMappings);
		pHostMappings = NULL;
	}

	fprintf(fp_Log, "[GpuFilterKernelStandalone] Initialization complete \n");
	fflush(fp_Log);

	return true;

}

void GpuFilterKernelStandalone::Process(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(), "GpuFilterKernelStandalone::In", fp_Log);

//	EvaluateEventsInCpu(_iNumEvents);
#endif

	if(!b_DeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuFilterKernelStandalone", fp_Log);
		b_DeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	// copy byte buffer
	p_InputEventBuffer->CopyToDevice(true);

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Invoke kernel Blocks(%d,%d) Threads(%d,%d) Events=%d\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1, _iNumEvents);
//	fprintf(fp_Log, "[GpuFilterKernelStandalone] InputBuf=%p Filter=%p Meta=%p ResultBuf=%p\n", p_InputEventBuffer->GetDeviceEventBuffer(),
//			p_DeviceFilter, p_InputEventBuffer->GetDeviceMetaEvent(), p_ResultEventBuffer->GetDeviceEventBuffer());
	fflush(fp_Log);
#endif

	ProcessEventsFilterKernelStandalone<<<numBlocks, numThreads>>>(
			p_DeviceParameters,
			_iNumEvents
	);

	if(b_LastKernel)
	{
		p_ResultEventBuffer->CopyToHost(false);
		fprintf(fp_Log, "[GpuFilterKernelStandalone] p_ResultEventBuffer %d\n", p_ResultEventBuffer->GetHostEventBuffer()[0]);
		fflush(fp_Log);
	}

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_DEBUG
	fprintf(fp_Log, "[GpuFilterKernelStandalone] ProcessEventsFilterKernelStandalone results\n");
	int * pResults = p_ResultEventBuffer->GetHostEventBuffer();
	for(int i=0; i<_iNumEvents; ++i)
	{
		fprintf(fp_Log, "[GpuFilterKernelStandalone] Result [%d => %d] \n", i, *pResults);
		pResults++;
	}
	fflush(fp_Log);
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Kernel complete \n");
	fflush(fp_Log);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[GpuFilterKernelStandalone] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif
}

char * GpuFilterKernelStandalone::GetResultEventBuffer()
{
	return (char*)p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuFilterKernelStandalone::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

void GpuFilterKernelStandalone::EvaluateEventsInCpu(int _iNumEvents)
{
	fprintf(fp_Log, "EvaluateEvenetsInCpu [NumEvents=%d] \n", _iNumEvents);

	GpuMetaEvent * pEventMeta = p_InputEventBuffer->GetHostMetaEvent();
	char * pBuffer = p_InputEventBuffer->GetHostEventBuffer();

	fprintf(fp_Log, "EventMeta %d [", pEventMeta->i_AttributeCount);
	for(int i=0; i<pEventMeta->i_AttributeCount; ++i)
	{
		fprintf(fp_Log, "Pos=%d,Type=%d,Len=%d|",
				pEventMeta->p_Attributes[i].i_Position,
				pEventMeta->p_Attributes[i].i_Type,
				pEventMeta->p_Attributes[i].i_Length);
	}
	fprintf(fp_Log, "]\n");
	fflush(fp_Log);

	// get assigned filter
	GpuFilterProcessor * pFilter = (GpuFilterProcessor*)p_Processor;
	pFilter->Print(fp_Log);

	int iNumBlocks = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);

	fprintf(fp_Log, "EvaluateEvenetsInCpu [Blocks=%d|ThreadsPerBlock=%d] \n", iNumBlocks, i_ThreadBlockSize);
	fflush(fp_Log);

	for(int blockidx=0; blockidx<iNumBlocks; ++blockidx)
	{
		for(int threadidx=0; threadidx<i_ThreadBlockSize; ++threadidx)
		{
			if((blockidx == _iNumEvents / i_ThreadBlockSize) && // last thread block
					(threadidx >= _iNumEvents % i_ThreadBlockSize))
			{
				continue;
			}

			// get assigned event
			int iEventIdx = (blockidx * i_ThreadBlockSize) +  threadidx;
			char * pEvent = pBuffer + (pEventMeta->i_SizeOfEventInBytes * iEventIdx);

			fprintf(fp_Log, "Event_%d <%p> ", iEventIdx, pEvent);

			for(int a=0; a<pEventMeta->i_AttributeCount; ++a)
			{
				switch(pEventMeta->p_Attributes[a].i_Type)
				{
				case DataType::Boolean:
				{
					int16_t i;
					memcpy(&i, pEvent + pEventMeta->p_Attributes[a].i_Position, 2);
					fprintf(fp_Log, "[Bool|Pos=%d|Len=2|Val=%d] ", pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Int:
				{
					int32_t i;
					memcpy(&i, pEvent + pEventMeta->p_Attributes[a].i_Position, 4);
					fprintf(fp_Log, "[Int|Pos=%d|Len=4|Val=%d] ", pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Long:
				{
					int64_t i;
					memcpy(&i, pEvent + pEventMeta->p_Attributes[a].i_Position, 8);
					fprintf(fp_Log, "[Long|Pos=%d|Len=8|Val=%" PRIi64 "] ", pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Float:
				{
					float f;
					memcpy(&f, pEvent + pEventMeta->p_Attributes[a].i_Position, 4);
					fprintf(fp_Log, "[Float|Pos=%d|Len=4|Val=%f] ", pEventMeta->p_Attributes[a].i_Position, f);
				}
				break;
				case DataType::Double:
				{
					double f;
					memcpy(&f, pEvent + pEventMeta->p_Attributes[a].i_Position, 8);
					fprintf(fp_Log, "[Double|Pos=%d|Len=8|Val=%f] ", pEventMeta->p_Attributes[a].i_Position, f);
				}
				break;
				case DataType::StringIn:
				{
					int16_t i;
					memcpy(&i, pEvent + pEventMeta->p_Attributes[a].i_Position, 2);
					char * z = pEvent + pEventMeta->p_Attributes[a].i_Position + 2;
					z[i] = 0;
					fprintf(fp_Log, "[String|Pos=%d|Len=%d|Val=%s] ", pEventMeta->p_Attributes[a].i_Position, i, z);
				}
				break;
				default:
					break;
				}
			}

			fprintf(fp_Log, "\n");
			fflush(fp_Log);

			SiddhiCpu::FilterEvalParameters mEval;
			mEval.p_Filter = pFilter;
			mEval.p_Meta = pEventMeta;
			mEval.p_Event = pEvent;
			mEval.i_CurrentIndex = 0;
			mEval.fp_Log = fp_Log;

			bool bResult = SiddhiCpu::Evaluate(mEval);
			fflush(fp_Log);

			if(bResult)
			{
				fprintf(fp_Log, "Matched [%d] \n", iEventIdx);
			}
			else // ~ possible way to avoid cudaMemset from host
			{
				fprintf(fp_Log, "Not Matched [%d] \n", iEventIdx);
			}

			fflush(fp_Log);
		}
	}
}

// ============================================================================================================

GpuFilterKernelFirst::GpuFilterKernelFirst(GpuProcessor * _pProc, GpuProcessorContext * _pContext, int _iThreadBlockSize, FILE * _fPLog) :
	GpuKernel(_pProc, _pContext->GetDeviceId(), _iThreadBlockSize, _fPLog),
	p_Context(_pContext),
	p_DeviceFilter(NULL),
	p_InputEventBuffer(NULL),
	p_MatchedIndexEventBuffer(NULL),
	p_PrefixSumBuffer(NULL),
	p_ResultEventBuffer(NULL),
	i_MatchedEvenBufferIndex(-1),
	p_TempStorageForPrefixSum(NULL),
	i_SizeOfTempStorageForPrefixSum(0),
	p_DeviceParameters(NULL),
	b_DeviceSet(false)
{

}

GpuFilterKernelFirst::~GpuFilterKernelFirst()
{
	fprintf(fp_Log, "[GpuFilterKernelFirst] destroy\n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaFree(p_DeviceFilter));
	p_DeviceFilter = NULL;

	delete p_ResultEventBuffer;
	p_ResultEventBuffer = NULL;

	delete p_MatchedIndexEventBuffer;
	p_MatchedIndexEventBuffer = NULL;

	delete p_PrefixSumBuffer;
	p_PrefixSumBuffer = NULL;

	CUDA_CHECK_RETURN(cudaFree(p_TempStorageForPrefixSum));
	p_TempStorageForPrefixSum = NULL;

	if(p_DeviceOutputAttributeMapping)
	{
		CUDA_CHECK_RETURN(cudaFree(p_DeviceOutputAttributeMapping));
		p_DeviceOutputAttributeMapping = NULL;
	}

	CUDA_CHECK_RETURN(cudaFree(p_DeviceParameters));
	p_DeviceParameters = NULL;

//	sdkDeleteTimer(&p_StopWatch);
//	p_StopWatch = NULL;
}

bool GpuFilterKernelFirst::Initialize(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize)
{
	fprintf(fp_Log, "[GpuFilterKernelFirst] Initialize : StreamIndex=%d\n", _iStreamIndex);
	fflush(fp_Log);

	// set input event buffer
	fprintf(fp_Log, "[GpuFilterKernelFirst] InputEventBufferIndex=%d\n", i_InputBufferIndex);
	fflush(fp_Log);
	p_InputEventBuffer = (GpuStreamEventBuffer*)p_Context->GetEventBuffer(i_InputBufferIndex);
	p_InputEventBuffer->Print();

	// set resulting event buffer and its meta data
	GpuMetaEvent * pMatchedResultIndexMetaEvent = new GpuMetaEvent(_pMetaEvent->i_StreamIndex, 1, sizeof(int));
	pMatchedResultIndexMetaEvent->SetAttribute(0, DataType::Int, sizeof(int), 0);

	p_MatchedIndexEventBuffer = new GpuIntBuffer("FilterMatchedIndexEventBuffer", p_Context->GetDeviceId(), pMatchedResultIndexMetaEvent, fp_Log);
	p_MatchedIndexEventBuffer->CreateEventBuffer(_iInputEventBufferSize);
	fprintf(fp_Log, "[GpuFilterKernelFirst] MatchedIndexEventBuffer created : Size=%d bytes\n",
			p_MatchedIndexEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_MatchedIndexEventBuffer->Print();

	p_PrefixSumBuffer = new GpuIntBuffer("FilterPrefixSumBuffer", p_Context->GetDeviceId(), pMatchedResultIndexMetaEvent, fp_Log);
	p_PrefixSumBuffer->CreateEventBuffer(_iInputEventBufferSize);
	p_PrefixSumBuffer->Print();

	delete pMatchedResultIndexMetaEvent;

	i_SizeOfTempStorageForPrefixSum = sizeof(int) * 2 * _iInputEventBufferSize;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&p_TempStorageForPrefixSum, i_SizeOfTempStorageForPrefixSum));


	p_ResultEventBuffer = new GpuStreamEventBuffer("FilterResultEventBuffer", p_Context->GetDeviceId(), _pMetaEvent, fp_Log);
	p_ResultEventBuffer->CreateEventBuffer(_iInputEventBufferSize);

	i_ResultEventBufferIndex = p_Context->AddEventBuffer(p_ResultEventBuffer);

	fprintf(fp_Log, "[GpuFilterKernelFirst] ResultEventBuffer created : Index=%d Size=%d bytes\n", i_ResultEventBufferIndex,
			p_ResultEventBuffer->GetEventBufferSizeInBytes());
	fflush(fp_Log);
	p_ResultEventBuffer->Print();

	fprintf(fp_Log, "[GpuFilterKernelFirst] Copying filter to device \n");
	fflush(fp_Log);

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &p_DeviceFilter,
			sizeof(GpuKernelFilter)));

	GpuKernelFilter * apHostFilters = (GpuKernelFilter *) malloc(sizeof(GpuKernelFilter));

	GpuFilterProcessor * pFilter = (GpuFilterProcessor*)p_Processor;

	apHostFilters->i_NodeCount = pFilter->i_NodeCount;
	apHostFilters->ap_ExecutorNodes = NULL;

	CUDA_CHECK_RETURN(cudaMalloc(
			(void**) &apHostFilters->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount));

	CUDA_CHECK_RETURN(cudaMemcpy(
			apHostFilters->ap_ExecutorNodes,
			pFilter->ap_ExecutorNodes,
			sizeof(ExecutorNode) * pFilter->i_NodeCount,
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_DeviceFilter,
			apHostFilters,
			sizeof(GpuKernelFilter),
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	free(apHostFilters);
	apHostFilters = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &p_DeviceParameters, sizeof(KernelParametersFilterStandalone)));
	KernelParametersFilterStandalone * pHostParameters = (KernelParametersFilterStandalone*) malloc(sizeof(KernelParametersFilterStandalone));

	pHostParameters->p_InByteBuffer = p_InputEventBuffer->GetDeviceEventBuffer();
	pHostParameters->ap_Filter = p_DeviceFilter;
	pHostParameters->p_MetaEvent = p_InputEventBuffer->GetDeviceMetaEvent();
	pHostParameters->p_ResultBuffer = p_MatchedIndexEventBuffer->GetDeviceEventBuffer();
	pHostParameters->i_SizeOfEvent = p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes;
	pHostParameters->i_EventsPerBlock = i_ThreadBlockSize;

	CUDA_CHECK_RETURN(cudaMemcpy(
			p_DeviceParameters,
			pHostParameters,
			sizeof(KernelParametersFilterStandalone),
			cudaMemcpyHostToDevice));

	free(pHostParameters);
	pHostParameters = NULL;


	// copy Output mappings
	if(p_HostOutputAttributeMapping)
	{
		fprintf(fp_Log, "[GpuFilterKernelFirst] Copying AttributeMappings to device \n");
		fflush(fp_Log);

		fprintf(fp_Log, "[GpuFilterKernelFirst] AttributeMapCount : %d \n", p_HostOutputAttributeMapping->i_MappingCount);
		for(int c=0; c<p_HostOutputAttributeMapping->i_MappingCount; ++c)
		{
			fprintf(fp_Log, "[GpuFilterKernelFirst] Map : Form [Stream=%d, Attrib=%d] To [Attrib=%d] \n",
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::STREAM_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].from[AttributeMapping::ATTRIBUTE_INDEX],
					p_HostOutputAttributeMapping->p_Mappings[c].to);

		}

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &p_DeviceOutputAttributeMapping,
				sizeof(AttributeMappings)));

		AttributeMappings * pHostMappings = (AttributeMappings*) malloc(sizeof(AttributeMappings));
		pHostMappings->i_MappingCount = p_HostOutputAttributeMapping->i_MappingCount;
		pHostMappings->p_Mappings = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(
				(void**) &pHostMappings->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount));

		CUDA_CHECK_RETURN(cudaMemcpy(
				pHostMappings->p_Mappings,
				p_HostOutputAttributeMapping->p_Mappings,
				sizeof(AttributeMapping) * p_HostOutputAttributeMapping->i_MappingCount,
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(
				p_DeviceOutputAttributeMapping,
				pHostMappings,
				sizeof(AttributeMappings),
				cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaThreadSynchronize());

		free(pHostMappings);
		pHostMappings = NULL;
	}

	fprintf(fp_Log, "[GpuFilterKernelFirst] Initialization complete \n");
	fflush(fp_Log);

	return true;

}

void GpuFilterKernelFirst::Process(int _iStreamIndex, int & _iNumEvents)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	GpuUtils::PrintByteBuffer(p_InputEventBuffer->GetHostEventBuffer(), _iNumEvents, p_InputEventBuffer->GetHostMetaEvent(), "GpuFilterKernelFirst::In", fp_Log);
#endif

	if(!b_DeviceSet)
	{
		GpuCudaHelper::SelectDevice(i_DeviceId, "GpuFilterKernelFirst", fp_Log);
		b_DeviceSet = true;
	}

#ifdef KERNEL_TIME
	sdkStartTimer(&p_StopWatch);
#endif

	// copy byte buffer
	p_InputEventBuffer->CopyToDevice(true);

	// call entry kernel
	int numBlocksX = ceil((float)_iNumEvents / (float)i_ThreadBlockSize);
	int numBlocksY = 1;
	dim3 numBlocks = dim3(numBlocksX, numBlocksY);
	dim3 numThreads = dim3(i_ThreadBlockSize, 1);

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuFilterKernelFirst] Invoke kernel Blocks(%d,%d) Threads(%d,%d)\n", numBlocksX, numBlocksY, i_ThreadBlockSize, 1);
	fflush(fp_Log);
#endif

	ProcessEventsFilterKernelFirstV2<<<numBlocks, numThreads>>>(
			p_DeviceParameters,
			_iNumEvents
	);

	CUDA_CHECK_RETURN(cub::DeviceScan::ExclusiveSum(
			p_TempStorageForPrefixSum, i_SizeOfTempStorageForPrefixSum,
			p_MatchedIndexEventBuffer->GetDeviceEventBuffer(),
			p_PrefixSumBuffer->GetDeviceEventBuffer(),
			_iNumEvents));


//	char               * _pInByteBuffer,      // Input ByteBuffer from java side
//	int                * _pMatchedIndexBuffer,// Matched event index buffer
//	int                * _pPrefixSumBuffer,   // prefix sum buffer
//	int                  _iEventCount,        // Num events in original batch
//	int                  _iSizeOfEvent,       // Size of an event
//	int                  _iEventsPerBlock,    // number of events allocated per block
//	char               * _pOutputEventBuffer  // Matched events final buffer

	ResultSorter<<<numBlocks, numThreads>>>(
			p_InputEventBuffer->GetDeviceEventBuffer(),
			p_MatchedIndexEventBuffer->GetDeviceEventBuffer(),
			p_PrefixSumBuffer->GetDeviceEventBuffer(),
			_iNumEvents,
			p_InputEventBuffer->GetHostMetaEvent()->i_SizeOfEventInBytes,
			i_ThreadBlockSize,
			p_ResultEventBuffer->GetDeviceEventBuffer()
	);

	if(b_LastKernel)
	{
		p_ResultEventBuffer->CopyToHost(true);
	}

	p_PrefixSumBuffer->CopyToHost(true);

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	int * pPrefixSumResults = p_PrefixSumBuffer->GetHostEventBuffer();
	_iNumEvents = pPrefixSumResults[_iNumEvents - 1];

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	if(b_LastKernel)
	{
		GpuUtils::PrintByteBuffer(p_ResultEventBuffer->GetHostEventBuffer(), _iNumEvents, p_ResultEventBuffer->GetHostMetaEvent(),
				"GpuFilterKernelFirst::Out", fp_Log);
	}
#endif

#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(fp_Log, "[GpuFilterKernelFirst] Kernel complete : ResultCount=%d\n", _iNumEvents);
	fflush(fp_Log);
#endif

#ifdef KERNEL_TIME
	sdkStopTimer(&p_StopWatch);
	float fElapsed = sdkGetTimerValue(&p_StopWatch);
	fprintf(fp_Log, "[ProcessEvents] Stats : Elapsed=%f ms\n", fElapsed);
	fflush(fp_Log);
	lst_ElapsedTimes.push_back(fElapsed);
	sdkResetTimer(&p_StopWatch);
#endif
}

char * GpuFilterKernelFirst::GetResultEventBuffer()
{
	return (char*)p_ResultEventBuffer->GetHostEventBuffer();
}

int GpuFilterKernelFirst::GetResultEventBufferSize()
{
	return p_ResultEventBuffer->GetEventBufferSizeInBytes();
}

}


#endif
