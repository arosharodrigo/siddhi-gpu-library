/*
 * GpuUtils.cpp
 *
 *  Created on: Jan 21, 2015
 *      Author: prabodha
 */


#include "../../domain/GpuMetaEvent.h"
#include "../../util/GpuUtils.h"
#include "../../domain/DataTypes.h"
#include "../../domain/GpuKernelDataTypes.h"
#include <string.h>
#include <unistd.h>
#include <syscall.h>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace SiddhiGpu
{

void GpuUtils::PrintThreadInfo(const char * _zTag, FILE * _fpLog)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_TRACE
	pid_t pid = getpid();
	int tid = syscall(__NR_gettid);
	fprintf(_fpLog, "[%s] PrintThreadInfo : PID=%d TID=%d\n", _zTag, pid, tid);
#endif
}

void GpuUtils::PrintByteBuffer(char * _pEventBuffer, int _iNumEvents, GpuMetaEvent * _pEventMeta, const char * _zTag, FILE * _fpLog)
{
#if GPU_DEBUG >= GPU_DEBUG_LEVEL_INFO
	fprintf(_fpLog, "[%s] [PrintByteBuffer] EventMeta [Attribs=%d] [", _zTag, _pEventMeta->i_AttributeCount);
	for(int i=0; i<_pEventMeta->i_AttributeCount; ++i)
	{
		fprintf(_fpLog, "Pos=%d,Type=%d,Len=%d|",
				_pEventMeta->p_Attributes[i].i_Position,
				_pEventMeta->p_Attributes[i].i_Type,
				_pEventMeta->p_Attributes[i].i_Length);
	}
	fprintf(_fpLog, "]\n");


	fprintf(_fpLog, "[%s] [PrintByteBuffer] Events Count : %d\n", _zTag, _iNumEvents);
	for(int e=0; e<_iNumEvents; ++e)
	{
		int offset = (_pEventMeta->i_SizeOfEventInBytes * e);
		char * pEventBuffer = _pEventBuffer + offset;
		GpuEvent * pEvent = (GpuEvent*) pEventBuffer;

		fprintf(_fpLog, "[%s] [PrintByteBuffer] [%-5d] [%-5d] Event_%" PRIu64 " <%" PRIu64 "> Type=%u ", _zTag, e, offset, pEvent->i_Sequence, pEvent->i_Timestamp, pEvent->i_Type);

		if(pEvent->i_Type != GpuEvent::NONE)
		{
			for(int a=0; a<_pEventMeta->i_AttributeCount; ++a)
			{
				switch(_pEventMeta->p_Attributes[a].i_Type)
				{
				case DataType::Boolean:
				{
					int16_t i;
					memcpy(&i, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 2);
					fprintf(_fpLog, "[Bool|Pos=%d|Len=2|Val=%d] ", _pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Int:
				{
					int32_t i;
					memcpy(&i, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 4);
					fprintf(_fpLog, "[Int|Pos=%d|Len=4|Val=%d] ", _pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Long:
				{
					int64_t i;
					memcpy(&i, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 8);
					fprintf(_fpLog, "[Long|Pos=%d|Len=8|Val=%" PRIi64 "] ", _pEventMeta->p_Attributes[a].i_Position, i);
				}
				break;
				case DataType::Float:
				{
					float f;
					memcpy(&f, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 4);
					fprintf(_fpLog, "[Float|Pos=%d|Len=4|Val=%f] ", _pEventMeta->p_Attributes[a].i_Position, f);
				}
				break;
				case DataType::Double:
				{
					double f;
					memcpy(&f, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 8);
					fprintf(_fpLog, "[Double|Pos=%d|Len=8|Val=%f] ", _pEventMeta->p_Attributes[a].i_Position, f);
				}
				break;
				case DataType::StringIn:
				{
					int16_t i;
					memcpy(&i, pEventBuffer + _pEventMeta->p_Attributes[a].i_Position, 2);
					char * z = pEventBuffer + _pEventMeta->p_Attributes[a].i_Position + 2;
					z[i] = 0;
					fprintf(_fpLog, "[String|Pos=%d|Len=%d|Val=%s] ", _pEventMeta->p_Attributes[a].i_Position, i, z);
				}
				break;
				default:
					break;
				}
			}
		}

		fprintf(_fpLog, "\n");
		fflush(_fpLog);
	}
#endif
}

}
