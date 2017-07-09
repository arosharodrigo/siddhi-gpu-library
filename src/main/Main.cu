#include <stdio.h>
#include "../util/GpuCudaHelper.h"
#include "../util/GpuUtils.h"

int main(void)
{
	float maxError = 0.5f;
	printf("Max error: %f\n", maxError);

	FILE * fp_Log;
	char zLogFile[256];
	sprintf(zLogFile, "/home/arosha/projects/Siddhi/projects/start-2017-06-19/NSIGHT/logs/info.log");
	fp_Log = fopen(zLogFile, "w");

	const char * _zTag = "tag1";
	fprintf(fp_Log, "[GpuCudaHelper] <%s> Selecting CUDA device\n", _zTag);

	int _iDeviceId = 0;
	SiddhiGpu::GpuCudaHelper helper;
	bool b = helper.SelectDevice(_iDeviceId, "GpuStreamProcessor::Configure", fp_Log);
	printf("isSelected: %d\n", b);

	SiddhiGpu::GpuUtils gpuUtils;
	gpuUtils.PrintThreadInfo("Hi there I am...", fp_Log);

}
