#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <forward_list>
#include <functional>
#include <initializer_list>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace scp
{
	template<typename TBase, typename TBuffer> class BigInt;
	template<typename TValue> class Frac;
	template<typename TInteger> class Rational;
	template<typename TValue> class Quat;


	struct TensorPosition;
	class TensorIteratorBase;
	class TensorShaped;

	class DynTensorIterator;
	template<uint64_t Order> class TensorIterator;

	class TensorSweepBase;
	class DynTensorSweep;
	template<uint64_t Order> class TensorSweep;
	template<uint64_t... Shape> class StaticTensorSweep;

	enum class ConvolutionMethod;
	template<typename TValue> class TensorBase;
	template<typename T> concept TensorConcept = requires { typename T::ValueType; } && std::derived_from<T, TensorBase<typename T::ValueType>>;

	template<TensorConcept TTensor> class MatrixBase;
	template<typename T> concept MatrixConcept = requires { typename T::TensorType; } && std::derived_from<T, MatrixBase<typename T::TensorType>>;
	template<TensorConcept TTensor> class VectorBase;
	template<typename T> concept VectorConcept = requires { typename T::MatrixType; } && std::derived_from<T, VectorBase<typename T::MatrixType>>;

	template<typename TValue> class SpTensor;
	template<typename TValue> class SpMatrix;
	template<typename TValue> class SpVector;

	template<typename TValue> class DenseTensor;
	template<typename T> concept DenseTensorConcept = TensorConcept<T> && std::derived_from<T, DenseTensor<typename T::ValueType>>;
	template<DenseTensorConcept TTensor> class DenseMatrix;
	template<DenseTensorConcept TTensor> class DenseVector;

	template<typename TValue> class DynTensor;
	template<typename TValue> class DynMatrix;
	template<typename TValue> class DynVector;

	template<typename TValue, uint64_t Order> class Tensor;
	template<typename TValue> class Matrix;
	template<typename TValue> class Vector;

	template<typename TValue, uint64_t... Shape> class StaticTensor;
	template<typename TValue, uint64_t NRow, uint64_t NCol> class StaticMatrix;
	template<typename TValue, uint64_t Size> class StaticVector;


	template <typename TNode, typename TEdge> class Graph;
}
