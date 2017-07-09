/*
 * GpuKernel.cpp
 *
 *  Created on: Jan 19, 2015
 *      Author: prabodha
 */

#include "../../main/GpuProcessor.h"
#include "../../main/GpuKernel.h"

namespace SiddhiGpu
{

GpuKernel::GpuKernel(GpuProcessor * _pProc, int _iCudaDeviceId, int _iThreadBlockSize, FILE * _fPLog) :
	i_DeviceId(_iCudaDeviceId),
	i_ThreadBlockSize(_iThreadBlockSize),
	i_ResultEventBufferIndex(-1),
	i_InputBufferIndex(-1),
	p_OutputStreamMeta(NULL),
	p_HostOutputAttributeMapping(NULL),
	p_DeviceOutputAttributeMapping(NULL),
	b_LastKernel(false),
	p_Processor(_pProc),
	fp_Log(_fPLog)
{

}

GpuKernel::~GpuKernel()
{
}

};


