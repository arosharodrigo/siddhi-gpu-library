
#include <stdio.h>
#include "../util/GpuCudaHelper.h"
#include "../util/GpuUtils.h"

#include "../main/GpuQueryRuntime.h"
#include "../domain/GpuMetaEvent.h"
#include "../window/GpuLengthSlidingWindowProcessor.h"
#include "../domain/DataTypes.h"
#include "../main/GpuStreamProcessor.h"

using namespace SiddhiGpu;

int main() {

	printf("Starting Window Sample ...\n");

	SiddhiGpu::GpuMetaEvent gpuMetaEvent (0, 1, 4);
//	gpuMetaEvent.SetAttribute(0,5,8,0);
	gpuMetaEvent.SetAttribute(0,0,4,0);

	SiddhiGpu::GpuLengthSlidingWindowProcessor gpuLengthSlidingWindowProcessor (2);

	AttributeMappings attributeMappings (1);
	attributeMappings.AddMapping(0, 0, 0, 0);
	gpuLengthSlidingWindowProcessor.SetOutputStream(&gpuMetaEvent, &attributeMappings);

	SiddhiGpu::GpuQueryRuntime gpuQueryRuntime ("windowQuery1", 0, 2);
	gpuQueryRuntime.AddStream("windowStream1", &gpuMetaEvent);
	gpuQueryRuntime.AddProcessor("windowStream1", &gpuLengthSlidingWindowProcessor);
	gpuQueryRuntime.Configure();

	char* windowStream1 = gpuQueryRuntime.GetInputEventBuffer("windowStream1");
		windowStream1[0] = '\x0';
		windowStream1[1] = '\x3';
		windowStream1[2] = '\x0';
		windowStream1[3] = '\x0';
		windowStream1[4] = '\x0';
		windowStream1[5] = '\x6';
		windowStream1[6] = '\x0';
		windowStream1[7] = '\x0';
		windowStream1[8] = '\0';

		GpuStreamProcessor * gpuStreamProcessor = gpuQueryRuntime.GetStream("windowStream1");
		int outputSize = gpuStreamProcessor->Process(2);
		char* resultEventBuffer = gpuLengthSlidingWindowProcessor.GetResultEventBuffer();
		int resultEventBufferSize = gpuLengthSlidingWindowProcessor.GetResultEventBufferSize();

		printf("Output Event Size: %d\n", outputSize);
		for(int i = 0; i < resultEventBufferSize ; i++) {
			printf("resultEventBuffer[%d]: %d\n", i, resultEventBuffer[i]);
		}
		printf("resultEventBufferSize: %d\n", resultEventBufferSize);

	return 0;
}
