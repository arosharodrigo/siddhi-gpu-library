/*
 * GpuFilterProcessor.h
 *
 *  Created on: Jan 20, 2015
 *      Author: prabodha
 */

#ifndef GPUFILTERPROCESSOR_H_
#define GPUFILTERPROCESSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "../main/GpuProcessor.h"
#include "../domain/DataTypes.h"
#include "../domain/Value.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace SiddhiGpu
{

class GpuFilterKernelStandalone;

#pragma pack(1)

enum ConditionType
{
	EXECUTOR_NOOP = 0,

	EXECUTOR_AND,
	EXECUTOR_OR,
	EXECUTOR_NOT,
	EXECUTOR_BOOL,

	EXECUTOR_EQ_BOOL_BOOL,
	EXECUTOR_EQ_INT_INT,
	EXECUTOR_EQ_INT_LONG,
	EXECUTOR_EQ_INT_FLOAT,
	EXECUTOR_EQ_INT_DOUBLE,
	EXECUTOR_EQ_LONG_INT,
	EXECUTOR_EQ_LONG_LONG,
	EXECUTOR_EQ_LONG_FLOAT,
	EXECUTOR_EQ_LONG_DOUBLE,
	EXECUTOR_EQ_FLOAT_INT,
	EXECUTOR_EQ_FLOAT_LONG,
	EXECUTOR_EQ_FLOAT_FLOAT,
	EXECUTOR_EQ_FLOAT_DOUBLE,
	EXECUTOR_EQ_DOUBLE_INT,
	EXECUTOR_EQ_DOUBLE_LONG,
	EXECUTOR_EQ_DOUBLE_FLOAT,
	EXECUTOR_EQ_DOUBLE_DOUBLE,
	EXECUTOR_EQ_STRING_STRING,

	EXECUTOR_NE_BOOL_BOOL,
	EXECUTOR_NE_INT_INT,
	EXECUTOR_NE_INT_LONG,
	EXECUTOR_NE_INT_FLOAT,
	EXECUTOR_NE_INT_DOUBLE,
	EXECUTOR_NE_LONG_INT,
	EXECUTOR_NE_LONG_LONG,
	EXECUTOR_NE_LONG_FLOAT,
	EXECUTOR_NE_LONG_DOUBLE,
	EXECUTOR_NE_FLOAT_INT,
	EXECUTOR_NE_FLOAT_LONG,
	EXECUTOR_NE_FLOAT_FLOAT,
	EXECUTOR_NE_FLOAT_DOUBLE,
	EXECUTOR_NE_DOUBLE_INT,
	EXECUTOR_NE_DOUBLE_LONG,
	EXECUTOR_NE_DOUBLE_FLOAT,
	EXECUTOR_NE_DOUBLE_DOUBLE,
	EXECUTOR_NE_STRING_STRING,

	EXECUTOR_GT_INT_INT,
	EXECUTOR_GT_INT_LONG,
	EXECUTOR_GT_INT_FLOAT,
	EXECUTOR_GT_INT_DOUBLE,
	EXECUTOR_GT_LONG_INT,
	EXECUTOR_GT_LONG_LONG,
	EXECUTOR_GT_LONG_FLOAT,
	EXECUTOR_GT_LONG_DOUBLE,
	EXECUTOR_GT_FLOAT_INT,
	EXECUTOR_GT_FLOAT_LONG,
	EXECUTOR_GT_FLOAT_FLOAT,
	EXECUTOR_GT_FLOAT_DOUBLE,
	EXECUTOR_GT_DOUBLE_INT,
	EXECUTOR_GT_DOUBLE_LONG,
	EXECUTOR_GT_DOUBLE_FLOAT,
	EXECUTOR_GT_DOUBLE_DOUBLE,

	EXECUTOR_LT_INT_INT,
	EXECUTOR_LT_INT_LONG,
	EXECUTOR_LT_INT_FLOAT,
	EXECUTOR_LT_INT_DOUBLE,
	EXECUTOR_LT_LONG_INT,
	EXECUTOR_LT_LONG_LONG,
	EXECUTOR_LT_LONG_FLOAT,
	EXECUTOR_LT_LONG_DOUBLE,
	EXECUTOR_LT_FLOAT_INT,
	EXECUTOR_LT_FLOAT_LONG,
	EXECUTOR_LT_FLOAT_FLOAT,
	EXECUTOR_LT_FLOAT_DOUBLE,
	EXECUTOR_LT_DOUBLE_INT,
	EXECUTOR_LT_DOUBLE_LONG,
	EXECUTOR_LT_DOUBLE_FLOAT,
	EXECUTOR_LT_DOUBLE_DOUBLE,

	EXECUTOR_GE_INT_INT,
	EXECUTOR_GE_INT_LONG,
	EXECUTOR_GE_INT_FLOAT,
	EXECUTOR_GE_INT_DOUBLE,
	EXECUTOR_GE_LONG_INT,
	EXECUTOR_GE_LONG_LONG,
	EXECUTOR_GE_LONG_FLOAT,
	EXECUTOR_GE_LONG_DOUBLE,
	EXECUTOR_GE_FLOAT_INT,
	EXECUTOR_GE_FLOAT_LONG,
	EXECUTOR_GE_FLOAT_FLOAT,
	EXECUTOR_GE_FLOAT_DOUBLE,
	EXECUTOR_GE_DOUBLE_INT,
	EXECUTOR_GE_DOUBLE_LONG,
	EXECUTOR_GE_DOUBLE_FLOAT,
	EXECUTOR_GE_DOUBLE_DOUBLE,

	EXECUTOR_LE_INT_INT,
	EXECUTOR_LE_INT_LONG,
	EXECUTOR_LE_INT_FLOAT,
	EXECUTOR_LE_INT_DOUBLE,
	EXECUTOR_LE_LONG_INT,
	EXECUTOR_LE_LONG_LONG,
	EXECUTOR_LE_LONG_FLOAT,
	EXECUTOR_LE_LONG_DOUBLE,
	EXECUTOR_LE_FLOAT_INT,
	EXECUTOR_LE_FLOAT_LONG,
	EXECUTOR_LE_FLOAT_FLOAT,
	EXECUTOR_LE_FLOAT_DOUBLE,
	EXECUTOR_LE_DOUBLE_INT,
	EXECUTOR_LE_DOUBLE_LONG,
	EXECUTOR_LE_DOUBLE_FLOAT,
	EXECUTOR_LE_DOUBLE_DOUBLE,

	EXECUTOR_CONTAINS,

	EXECUTOR_INVALID, // set this for const and var nodes

	EXECUTOR_CONDITION_COUNT
};

inline const char * GetConditionName(ConditionType _eType)
{
	switch(_eType)
	{
	case EXECUTOR_NOOP:
		return "NOOP";
	case EXECUTOR_AND:
		return "AND";
	case EXECUTOR_OR:
		return "OR";
	case EXECUTOR_NOT:
		return "NOT";
	case EXECUTOR_BOOL:
		return "BOOL";
	case EXECUTOR_EQ_BOOL_BOOL:
		return "EQ_BOOL_BOOL";
	case EXECUTOR_EQ_INT_INT:
		return "EQ_INT_INT";
	case EXECUTOR_EQ_INT_LONG:
		return "EQ_INT_LONG";
	case EXECUTOR_EQ_INT_FLOAT:
		return "EQ_INT_FLOAT";
	case EXECUTOR_EQ_INT_DOUBLE:
		return "EQ_INT_DOUBLE";
	case EXECUTOR_EQ_LONG_INT:
		return "EQ_LONG_INT";
	case EXECUTOR_EQ_LONG_LONG:
		return "EQ_LONG_LONG";
	case EXECUTOR_EQ_LONG_FLOAT:
		return "EQ_LONG_FLOAT";
	case EXECUTOR_EQ_LONG_DOUBLE:
		return "EQ_LONG_DOUBLE";
	case EXECUTOR_EQ_FLOAT_INT:
		return "EQ_FLOAT_INT";
	case EXECUTOR_EQ_FLOAT_LONG:
		return "EQ_FLOAT_LONG";
	case EXECUTOR_EQ_FLOAT_FLOAT:
		return "EQ_FLOAT_FLOAT";
	case EXECUTOR_EQ_FLOAT_DOUBLE:
		return "EQ_FLOAT_DOUBLE";
	case EXECUTOR_EQ_DOUBLE_INT:
		return "EQ_DOUBLE_INT";
	case EXECUTOR_EQ_DOUBLE_LONG:
		return "EQ_DOUBLE_LONG";
	case EXECUTOR_EQ_DOUBLE_FLOAT:
		return "EQ_DOUBLE_FLOAT";
	case EXECUTOR_EQ_DOUBLE_DOUBLE:
		return "EQ_DOUBLE_DOUBLE";
	case EXECUTOR_EQ_STRING_STRING:
		return "EQ_STRING_STRING";
	case EXECUTOR_NE_BOOL_BOOL:
		return "NE_BOOL_BOOL";
	case EXECUTOR_NE_INT_INT:
		return "NE_INT_INT";
	case EXECUTOR_NE_INT_LONG:
		return "NE_INT_LONG";
	case EXECUTOR_NE_INT_FLOAT:
		return "NE_INT_FLOAT";
	case EXECUTOR_NE_INT_DOUBLE:
		return "NE_INT_DOUBLE";
	case EXECUTOR_NE_LONG_INT:
		return "NE_LONG_INT";
	case EXECUTOR_NE_LONG_LONG:
		return "NE_LONG_LONG";
	case EXECUTOR_NE_LONG_FLOAT:
		return "NE_LONG_FLOAT";
	case EXECUTOR_NE_LONG_DOUBLE:
		return "NE_LONG_DOUBLE";
	case EXECUTOR_NE_FLOAT_INT:
		return "NE_FLOAT_INT";
	case EXECUTOR_NE_FLOAT_LONG:
		return "NE_FLOAT_LONG";
	case EXECUTOR_NE_FLOAT_FLOAT:
		return "NE_FLOAT_FLOAT";
	case EXECUTOR_NE_FLOAT_DOUBLE:
		return "NE_FLOAT_DOUBLE";
	case EXECUTOR_NE_DOUBLE_INT:
		return "NE_DOUBLE_INT";
	case EXECUTOR_NE_DOUBLE_LONG:
		return "NE_DOUBLE_LONG";
	case EXECUTOR_NE_DOUBLE_FLOAT:
		return "NE_DOUBLE_FLOAT";
	case EXECUTOR_NE_DOUBLE_DOUBLE:
		return "NE_DOUBLE_DOUBLE";
	case EXECUTOR_NE_STRING_STRING:
		return "NE_STRING_STRING";
	case EXECUTOR_GT_INT_INT:
		return "GT_INT_INT";
	case EXECUTOR_GT_INT_LONG:
		return "GT_INT_LONG";
	case EXECUTOR_GT_INT_FLOAT:
		return "GT_INT_FLOAT";
	case EXECUTOR_GT_INT_DOUBLE:
		return "GT_INT_DOUBLE";
	case EXECUTOR_GT_LONG_INT:
		return "GT_LONG_INT";
	case EXECUTOR_GT_LONG_LONG:
		return "GT_LONG_LONG";
	case EXECUTOR_GT_LONG_FLOAT:
		return "GT_LONG_FLOAT";
	case EXECUTOR_GT_LONG_DOUBLE:
		return "GT_LONG_DOUBLE";
	case EXECUTOR_GT_FLOAT_INT:
		return "GT_FLOAT_INT";
	case EXECUTOR_GT_FLOAT_LONG:
		return "GT_FLOAT_LONG";
	case EXECUTOR_GT_FLOAT_FLOAT:
		return "GT_FLOAT_FLOAT";
	case EXECUTOR_GT_FLOAT_DOUBLE:
		return "GT_FLOAT_DOUBLE";
	case EXECUTOR_GT_DOUBLE_INT:
		return "GT_DOUBLE_INT";
	case EXECUTOR_GT_DOUBLE_LONG:
		return "GT_DOUBLE_LONG";
	case EXECUTOR_GT_DOUBLE_FLOAT:
		return "GT_DOUBLE_FLOAT";
	case EXECUTOR_GT_DOUBLE_DOUBLE:
		return "GT_DOUBLE_DOUBLE";
	case EXECUTOR_LT_INT_INT:
		return "LT_INT_INT";
	case EXECUTOR_LT_INT_LONG:
		return "LT_INT_LONG";
	case EXECUTOR_LT_INT_FLOAT:
		return "LT_INT_FLOAT";
	case EXECUTOR_LT_INT_DOUBLE:
		return "LT_INT_DOUBLE";
	case EXECUTOR_LT_LONG_INT:
		return "LT_LONG_INT";
	case EXECUTOR_LT_LONG_LONG:
		return "LT_LONG_LONG";
	case EXECUTOR_LT_LONG_FLOAT:
		return "LT_LONG_FLOAT";
	case EXECUTOR_LT_LONG_DOUBLE:
		return "LT_LONG_DOUBLE";
	case EXECUTOR_LT_FLOAT_INT:
		return "LT_FLOAT_INT";
	case EXECUTOR_LT_FLOAT_LONG:
		return "LT_FLOAT_LONG";
	case EXECUTOR_LT_FLOAT_FLOAT:
		return "LT_FLOAT_FLOAT";
	case EXECUTOR_LT_FLOAT_DOUBLE:
		return "LT_FLOAT_DOUBLE";
	case EXECUTOR_LT_DOUBLE_INT:
		return "LT_DOUBLE_INT";
	case EXECUTOR_LT_DOUBLE_LONG:
		return "LT_DOUBLE_LONG";
	case EXECUTOR_LT_DOUBLE_FLOAT:
		return "LT_DOUBLE_FLOAT";
	case EXECUTOR_LT_DOUBLE_DOUBLE:
		return "LT_DOUBLE_DOUBLE";
	case EXECUTOR_GE_INT_INT:
		return "GE_INT_INT";
	case EXECUTOR_GE_INT_LONG:
		return "GE_INT_LONG";
	case EXECUTOR_GE_INT_FLOAT:
		return "GE_INT_FLOAT";
	case EXECUTOR_GE_INT_DOUBLE:
		return "GE_INT_DOUBLE";
	case EXECUTOR_GE_LONG_INT:
		return "GE_LONG_INT";
	case EXECUTOR_GE_LONG_LONG:
		return "GE_LONG_LONG";
	case EXECUTOR_GE_LONG_FLOAT:
		return "GE_LONG_FLOAT";
	case EXECUTOR_GE_LONG_DOUBLE:
		return "GE_LONG_DOUBLE";
	case EXECUTOR_GE_FLOAT_INT:
		return "GE_FLOAT_INT";
	case EXECUTOR_GE_FLOAT_LONG:
		return "GE_FLOAT_LONG";
	case EXECUTOR_GE_FLOAT_FLOAT:
		return "GE_FLOAT_FLOAT";
	case EXECUTOR_GE_FLOAT_DOUBLE:
		return "GE_FLOAT_DOUBLE";
	case EXECUTOR_GE_DOUBLE_INT:
		return "GE_DOUBLE_INT";
	case EXECUTOR_GE_DOUBLE_LONG:
		return "GE_DOUBLE_LONG";
	case EXECUTOR_GE_DOUBLE_FLOAT:
		return "GE_DOUBLE_FLOAT";
	case EXECUTOR_GE_DOUBLE_DOUBLE:
		return "GE_DOUBLE_DOUBLE";
	case EXECUTOR_LE_INT_INT:
		return "LE_INT_INT";
	case EXECUTOR_LE_INT_LONG:
		return "LE_INT_LONG";
	case EXECUTOR_LE_INT_FLOAT:
		return "LE_INT_FLOAT";
	case EXECUTOR_LE_INT_DOUBLE:
		return "LE_INT_DOUBLE";
	case EXECUTOR_LE_LONG_INT:
		return "LE_LONG_INT";
	case EXECUTOR_LE_LONG_LONG:
		return "LE_LONG_LONG";
	case EXECUTOR_LE_LONG_FLOAT:
		return "LE_LONG_FLOAT";
	case EXECUTOR_LE_LONG_DOUBLE:
		return "LE_LONG_DOUBLE";
	case EXECUTOR_LE_FLOAT_INT:
		return "LE_FLOAT_INT";
	case EXECUTOR_LE_FLOAT_LONG:
		return "LE_FLOAT_LONG";
	case EXECUTOR_LE_FLOAT_FLOAT:
		return "LE_FLOAT_FLOAT";
	case EXECUTOR_LE_FLOAT_DOUBLE:
		return "LE_FLOAT_DOUBLE";
	case EXECUTOR_LE_DOUBLE_INT:
		return "LE_DOUBLE_INT";
	case EXECUTOR_LE_DOUBLE_LONG:
		return "LE_DOUBLE_LONG";
	case EXECUTOR_LE_DOUBLE_FLOAT:
		return "LE_DOUBLE_FLOAT";
	case EXECUTOR_LE_DOUBLE_DOUBLE:
		return "LE_DOUBLE_DOUBLE";
	case EXECUTOR_CONTAINS:
		return "CONTAINS";
	case EXECUTOR_INVALID:
		return "INVALID";
	default:
		break;
	}
	return "NONE";
}

enum ExpressionType
{
	EXPRESSION_CONST = 0,
	EXPRESSION_VARIABLE,
	EXPRESSION_TIME,
	EXPRESSION_SEQUENCE,

	EXPRESSION_ADD_INT,
	EXPRESSION_ADD_LONG,
	EXPRESSION_ADD_FLOAT,
	EXPRESSION_ADD_DOUBLE,

	EXPRESSION_SUB_INT,
	EXPRESSION_SUB_LONG,
	EXPRESSION_SUB_FLOAT,
	EXPRESSION_SUB_DOUBLE,

	EXPRESSION_MUL_INT,
	EXPRESSION_MUL_LONG,
	EXPRESSION_MUL_FLOAT,
	EXPRESSION_MUL_DOUBLE,

	EXPRESSION_DIV_INT,
	EXPRESSION_DIV_LONG,
	EXPRESSION_DIV_FLOAT,
	EXPRESSION_DIV_DOUBLE,

	EXPRESSION_MOD_INT,
	EXPRESSION_MOD_LONG,
	EXPRESSION_MOD_FLOAT,
	EXPRESSION_MOD_DOUBLE,

	EXPRESSION_INVALID,
	EXPRESSION_COUNT
};

inline const char * GetExpressionTypeName(ExpressionType _eType)
{
	switch(_eType)
	{
		case EXPRESSION_CONST:
			return "CONST";
		case EXPRESSION_VARIABLE:
			return "VARIABLE";
		case EXPRESSION_TIME:
			return "TIME";
		case EXPRESSION_SEQUENCE:
			return "SEQUENCE";
		case EXPRESSION_ADD_INT:
			return "ADD_INT";
		case EXPRESSION_ADD_LONG:
			return "ADD_LONG";
		case EXPRESSION_ADD_FLOAT:
			return "ADD_FLOAT";
		case EXPRESSION_ADD_DOUBLE:
			return "ADD_DOUBLE";
		case EXPRESSION_SUB_INT:
			return "SUB_INT";
		case EXPRESSION_SUB_LONG:
			return "SUB_LONG";
		case EXPRESSION_SUB_FLOAT:
			return "SUB_FLOAT";
		case EXPRESSION_SUB_DOUBLE:
			return "SUB_DOUBLE";
		case EXPRESSION_MUL_INT:
			return "MUL_INT";
		case EXPRESSION_MUL_LONG:
			return "MUL_LONG";
		case EXPRESSION_MUL_FLOAT:
			return "MUL_FLOAT";
		case EXPRESSION_MUL_DOUBLE:
			return "MUL_DOUBLE";
		case EXPRESSION_DIV_INT:
			return "DIV_INT";
		case EXPRESSION_DIV_LONG:
			return "DIV_LONG";
		case EXPRESSION_DIV_FLOAT:
			return "DIV_FLOAT";
		case EXPRESSION_DIV_DOUBLE:
			return "DIV_DOUBLE";
		case EXPRESSION_MOD_INT:
			return "MOD_INT";
		case EXPRESSION_MOD_LONG:
			return "MOD_LONG";
		case EXPRESSION_MOD_FLOAT:
			return "MOD_FLOAT";
		case EXPRESSION_MOD_DOUBLE:
			return "MOD_DOUBLE";
		default:
			break;
	}

	return "NONE";
}

enum ExecutorNodeType
{
	EXECUTOR_NODE_CONDITION = 0,
	EXECUTOR_NODE_EXPRESSION,

	EXECUTOR_NODE_TYPE_COUNT
};

inline const char * GetNodeTypeName(ExecutorNodeType _eType)
{
	switch(_eType)
	{
	case EXECUTOR_NODE_CONDITION:
		return "CONDITION";
	case EXECUTOR_NODE_EXPRESSION:
		return "EXPRESSION";
	default:
		break;
	}

	return "NONE";
}

class VariableValue
{
public:
	int i_StreamIndex;
	DataType::Value e_Type;
	int i_AttributePosition;

//	VariableValue();
//	VariableValue(int _iStreamIndex, DataType::Value _eType, int _iPos);

	VariableValue & Init();

	VariableValue & SetStreamIndex(int _iStreamIndex);
	VariableValue & SetDataType(DataType::Value _eType);
	VariableValue & SetPosition(int _iPos);

	void Print(FILE * _fp = stdout);
};

class ConstValue
{
public:
	DataType::Value e_Type;
	Values m_Value;

//	ConstValue();

	ConstValue & Init();

	ConstValue & SetBool(bool _bVal);
	ConstValue & SetInt(int _iVal);
	ConstValue & SetLong(int64_t _lVal);
	ConstValue & SetFloat(float _fval);
	ConstValue & SetDouble(double _dVal);
	ConstValue & SetString(const char * _zVal, int _iLen);

	void Print(FILE * _fp = stdout);
};

typedef struct ElementryNode
{
	int i_StreamIndex;
	DataType::Value e_Type;
	int i_AttributePosition;
	int i_OutputPosition;
}ElementryNode;

typedef struct ExpressionNode
{
	enum Operator
	{
		OP_ADD_INT = 0,
		OP_ADD_LONG,
		OP_ADD_FLOAT,
		OP_ADD_DOUBLE,

		OP_SUB_INT,
		OP_SUB_LONG,
		OP_SUB_FLOAT,
		OP_SUB_DOUBLE,

		OP_MUL_INT,
		OP_MUL_LONG,
		OP_MUL_FLOAT,
		OP_MUL_DOUBLE,

		OP_DIV_INT,
		OP_DIV_LONG,
		OP_DIV_FLOAT,
		OP_DIV_DOUBLE,

		OP_MOD_INT,
		OP_MOD_LONG,
		OP_MOD_FLOAT,
		OP_MOD_DOUBLE
	};

	Operator e_Operator;
	int i_LeftValuePos;
	int i_RightValuePos;
	int i_OutputPosition;
}ExpressionNode;

typedef struct ConditionNode
{
	enum Condition
	{
		COND_AND = 0,
		COND_OR,
		COND_NOT,
		COND_BOOL,

		COND_EQ_BOOL_BOOL,
		COND_EQ_INT_INT,
		COND_EQ_INT_LONG,
		COND_EQ_INT_FLOAT,
		COND_EQ_INT_DOUBLE,
		COND_EQ_LONG_INT,
		COND_EQ_LONG_LONG,
		COND_EQ_LONG_FLOAT,
		COND_EQ_LONG_DOUBLE,
		COND_EQ_FLOAT_INT,
		COND_EQ_FLOAT_LONG,
		COND_EQ_FLOAT_FLOAT,
		COND_EQ_FLOAT_DOUBLE,
		COND_EQ_DOUBLE_INT,
		COND_EQ_DOUBLE_LONG,
		COND_EQ_DOUBLE_FLOAT,
		COND_EQ_DOUBLE_DOUBLE,
		COND_EQ_STRING_STRING,

		COND_NE_BOOL_BOOL,
		COND_NE_INT_INT,
		COND_NE_INT_LONG,
		COND_NE_INT_FLOAT,
		COND_NE_INT_DOUBLE,
		COND_NE_LONG_INT,
		COND_NE_LONG_LONG,
		COND_NE_LONG_FLOAT,
		COND_NE_LONG_DOUBLE,
		COND_NE_FLOAT_INT,
		COND_NE_FLOAT_LONG,
		COND_NE_FLOAT_FLOAT,
		COND_NE_FLOAT_DOUBLE,
		COND_NE_DOUBLE_INT,
		COND_NE_DOUBLE_LONG,
		COND_NE_DOUBLE_FLOAT,
		COND_NE_DOUBLE_DOUBLE,
		COND_NE_STRING_STRING,

		COND_GT_INT_INT,
		COND_GT_INT_LONG,
		COND_GT_INT_FLOAT,
		COND_GT_INT_DOUBLE,
		COND_GT_LONG_INT,
		COND_GT_LONG_LONG,
		COND_GT_LONG_FLOAT,
		COND_GT_LONG_DOUBLE,
		COND_GT_FLOAT_INT,
		COND_GT_FLOAT_LONG,
		COND_GT_FLOAT_FLOAT,
		COND_GT_FLOAT_DOUBLE,
		COND_GT_DOUBLE_INT,
		COND_GT_DOUBLE_LONG,
		COND_GT_DOUBLE_FLOAT,
		COND_GT_DOUBLE_DOUBLE,

		COND_LT_INT_INT,
		COND_LT_INT_LONG,
		COND_LT_INT_FLOAT,
		COND_LT_INT_DOUBLE,
		COND_LT_LONG_INT,
		COND_LT_LONG_LONG,
		COND_LT_LONG_FLOAT,
		COND_LT_LONG_DOUBLE,
		COND_LT_FLOAT_INT,
		COND_LT_FLOAT_LONG,
		COND_LT_FLOAT_FLOAT,
		COND_LT_FLOAT_DOUBLE,
		COND_LT_DOUBLE_INT,
		COND_LT_DOUBLE_LONG,
		COND_LT_DOUBLE_FLOAT,
		COND_LT_DOUBLE_DOUBLE,

		COND_GE_INT_INT,
		COND_GE_INT_LONG,
		COND_GE_INT_FLOAT,
		COND_GE_INT_DOUBLE,
		COND_GE_LONG_INT,
		COND_GE_LONG_LONG,
		COND_GE_LONG_FLOAT,
		COND_GE_LONG_DOUBLE,
		COND_GE_FLOAT_INT,
		COND_GE_FLOAT_LONG,
		COND_GE_FLOAT_FLOAT,
		COND_GE_FLOAT_DOUBLE,
		COND_GE_DOUBLE_INT,
		COND_GE_DOUBLE_LONG,
		COND_GE_DOUBLE_FLOAT,
		COND_GE_DOUBLE_DOUBLE,

		COND_LE_INT_INT,
		COND_LE_INT_LONG,
		COND_LE_INT_FLOAT,
		COND_LE_INT_DOUBLE,
		COND_LE_LONG_INT,
		COND_LE_LONG_LONG,
		COND_LE_LONG_FLOAT,
		COND_LE_LONG_DOUBLE,
		COND_LE_FLOAT_INT,
		COND_LE_FLOAT_LONG,
		COND_LE_FLOAT_FLOAT,
		COND_LE_FLOAT_DOUBLE,
		COND_LE_DOUBLE_INT,
		COND_LE_DOUBLE_LONG,
		COND_LE_DOUBLE_FLOAT,
		COND_LE_DOUBLE_DOUBLE,

		COND_CONTAINS,
	};

	Condition e_Condition;
	int i_LeftValuePos;
	int i_RightValuePos;
	int i_OutputPosition;
}ConditionNode;

typedef struct ValueNode
{
	DataType::Value e_Type;
	Values m_Value;
}ValueNode;

class ExecutorNode
{
public:
	ExecutorNodeType e_NodeType;

	// if operator - what is the type
	ConditionType e_ConditionType;

	// if expression
	ExpressionType e_ExpressionType;

	// if const - what is the value
	ConstValue m_ConstValue;

	// if var - variable holder
	VariableValue m_VarValue;

	int i_ParentNodeIndex;
	bool b_Processed;

//	ExecutorNode();
	ExecutorNode & Init();

	ExecutorNode & SetNodeType(ExecutorNodeType _eNodeType);
	ExecutorNode & SetConditionType(ConditionType _eCondType);
	ExecutorNode & SetExpressionType(ExpressionType _eExprType);
	ExecutorNode & SetConstValue(ConstValue _mConstVal);
	ExecutorNode & SetVariableValue(VariableValue _mVarValue);

	ExecutorNode & SetParentNode(int _iParentIndex);

	void Print() { Print(stdout); }
	void Print(FILE * _fp);
};

#pragma pack()

class GpuFilterProcessor : public GpuProcessor
{
public:
	GpuFilterProcessor(int _iNodeCount);
	virtual ~GpuFilterProcessor();

	void AddExecutorNode(int _iPos, ExecutorNode & _pNode);

	void Destroy();

	void Configure(int _iStreamIndex, GpuProcessor * _pPrevProcessor, GpuProcessorContext * _pContext, FILE * _fpLog);
	void Init(int _iStreamIndex, GpuMetaEvent * _pMetaEvent, int _iInputEventBufferSize);
	int Process(int _iStreamIndex, int _iNumEvents);
	void Print(FILE * _fp);
	GpuProcessor * Clone();

	int GetResultEventBufferIndex();
	char * GetResultEventBuffer();
	int GetResultEventBufferSize();

	void Print() { Print(stdout); }

	ExecutorNode * ap_ExecutorNodes; // nodes are stored in in-order
	int            i_NodeCount;

private:
	GpuProcessorContext * p_Context;
	GpuKernel * p_FilterKernel;
};

};


#endif /* GPUFILTERPROCESSOR_H_ */
