
#include <stdio.h>
#include "../util/GpuCudaHelper.h"
#include "../util/GpuUtils.h"

#include "../main/GpuQueryRuntime.h"
#include "../domain/GpuMetaEvent.h"
#include "../filter/GpuFilterProcessor.h"
#include "../domain/DataTypes.h"
#include "../main/GpuStreamProcessor.h"

using namespace SiddhiGpu;

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
	c1.Init().SetInt(100);
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
	char* resultEventBuffer = gpuFilterProcessor.GetResultEventBuffer();
	int resultEventBufferSize = gpuFilterProcessor.GetResultEventBufferSize();

	printf("outputSize: %d\n", outputSize);
	printf("resultEventBuffer[0]: %d\n", resultEventBuffer[0]);
	printf("resultEventBuffer[1]: %d\n", resultEventBuffer[1]);
	printf("resultEventBuffer[2]: %d\n", resultEventBuffer[2]);
	printf("resultEventBuffer[3]: %d\n", resultEventBuffer[3]);
	printf("resultEventBuffer[4]: %d\n", resultEventBuffer[4]);
	printf("resultEventBufferSize: %d\n", resultEventBufferSize);

	return 0;
}
