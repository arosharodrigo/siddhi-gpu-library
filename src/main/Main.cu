#include <stdio.h>
#include "../util/GpuCudaHelper.h"
#include "../util/GpuUtils.h"

#include "../main/GpuQueryRuntime.h"
#include "../domain/GpuMetaEvent.h"
#include "../filter/GpuFilterProcessor.h"
#include "../domain/DataTypes.h"
#include "../main/GpuStreamProcessor.h"

using namespace SiddhiGpu;

/*int main(void)
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

}*/

int main() {

	SiddhiGpu::GpuMetaEvent gpuMetaEvent (0, 1, 4);
//	gpuMetaEvent.SetAttribute(0,5,8,0);
	gpuMetaEvent.SetAttribute(0,0,4,0);


	SiddhiGpu::GpuFilterProcessor gpuFilterProcessor (3);

	ExecutorNode node1;
	ExecutorNode node2;
	ExecutorNode node3;

	node1.Init();
	node1.SetNodeType(EXECUTOR_NODE_CONDITION);
	node1.SetConditionType(EXECUTOR_GE_INT_INT);

	VariableValue v1;
	v1.Init().SetPosition(0).SetStreamIndex(0).SetDataType(DataType::Int);
	node2.Init();
	node2.SetNodeType(EXECUTOR_NODE_EXPRESSION);
	node2.SetExpressionType(EXPRESSION_VARIABLE);
	node2.SetVariableValue(v1);
	node2.SetParentNode(0);

	ConstValue c1;
	c1.Init().SetInt(1000);
	node3.Init();
	node3.SetNodeType(EXECUTOR_NODE_EXPRESSION);
	node3.SetExpressionType(EXPRESSION_CONST);
	node3.SetConstValue(c1);
	node3.SetParentNode(0);

	gpuFilterProcessor.AddExecutorNode(0, node1);
	gpuFilterProcessor.AddExecutorNode(1, node2);
	gpuFilterProcessor.AddExecutorNode(2, node3);

	SiddhiGpu::GpuQueryRuntime gpuQueryRuntime ("filterQuery1", 0, 1);
	gpuQueryRuntime.AddStream("filterStream1", &gpuMetaEvent);
	gpuQueryRuntime.AddProcessor("filterStream1", &gpuFilterProcessor);
	gpuQueryRuntime.Configure();
	char* filterStream1 = gpuQueryRuntime.GetInputEventBuffer("filterStream1");
	filterStream1[0] = '\xE7';
	filterStream1[1] = '\x3';
	filterStream1[2] = '\x0';
	filterStream1[3] = '\x0';
	filterStream1[4] = '\0';


	GpuStreamProcessor * gpuStreamProcessor = gpuQueryRuntime.GetStream("filterStream1");
	int outputSize = gpuStreamProcessor->Process(1);
	printf("outputSize: %d", outputSize);

}
