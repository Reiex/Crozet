///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

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
#include <iterator>
#include <map>
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
	template<typename T> concept IsComplex = requires { typename T::value_type; } && std::same_as<std::complex<typename T::value_type>, T>;

	template<typename T1, typename T2, typename R> concept Addable = requires (T1 x1, T2 x2)
	{
		{ x1 + x2 } -> std::convertible_to<R>;
	};
	template<typename T1, typename T2, typename R> concept Substractable = requires (T1 x1, T2 x2)
	{
		{ x1 - x2 } -> std::convertible_to<R>;
	};
	template<typename T1, typename T2, typename R> concept Multipliable = requires (T1 x1, T2 x2)
	{
		{ x1* x2 } -> std::convertible_to<R>;
	};
	template<typename T1, typename T2, typename R> concept Divisible = requires (T1 x1, T2 x2)
	{
		{ x1 / x2 } -> std::convertible_to<R>;
	};

	template<typename T> concept Monoid =
		Multipliable<T, T, T>
		&& requires (T x)
	{
		x = 1;
	};
	template<typename T> concept CommutativeMonoid =
		Addable<T, T, T>
		&& requires (T x)
	{
		x = 0;
	};

	template<typename T> concept Group =
		Monoid<T>
		&& Divisible<T, T, T>;
	template<typename T> concept CommutativeGroup =
		CommutativeMonoid<T>
		&& Substractable<T, T, T>;

	template<typename T> concept Ring =
		CommutativeGroup<T>
		&& Monoid<T>;

	template<typename T> concept Field =
		Ring<T>
		&& Divisible<T, T, T>;

	template<typename T, typename R> concept Module =
		CommutativeGroup<T>
		&& Ring<R>
		&& Multipliable<T, R, T>
		&& Multipliable<R, T, T>
		&& Divisible<T, R, T>
		&& Divisible<R, T, T>;

	template<typename T, typename F> concept VectorSpace =
		Module<T, F>
		&& Field<F>;

	template<typename T, typename F> concept Algebra =
		VectorSpace<T, F>
		&& Monoid<T>;
	template<typename T, typename F> concept DivisionAlgebra =
		VectorSpace<T, F>
		&& Group<T>;


	template<typename TBase, typename TBuffer> class BigInt;
	template<typename TValue> class Frac;
	template<typename TInteger> class Rational;
	template<typename TValue> class Quat;


	struct TensorPosition;
	class TensorShapeIterator;
	struct TensorShape;
	enum class BorderBehaviour;
	enum class InterpolationMethod;
	template<typename T> concept UntypedTensorConcept = requires { typename T::ValueType; };	// TODO: Better than that
	template<typename T> concept UntypedMatrixConcept = requires { typename T::Super; typename T::IsMatrix; }&& UntypedTensorConcept<typename T::Super>;
	template<typename T> concept UntypedVectorConcept = requires { typename T::Super; typename T::IsVector; }&& UntypedTensorConcept<typename T::Super>;
	template<typename T, typename TValue> concept TensorConcept = UntypedTensorConcept<T> && std::same_as<TValue, typename T::ValueType>;
	template<typename T, typename TValue> concept MatrixConcept = UntypedMatrixConcept<T> && std::same_as<TValue, typename T::Super::ValueType>;
	template<typename T, typename TValue> concept VectorConcept = UntypedVectorConcept<T> && std::same_as<TValue, typename T::Super::ValueType>;

	template<typename TValue> class Tensor;
	template<typename TValue> class Matrix;
	template<typename TValue> class Vector;

	template<typename TValue> class SpTensor;
	template<typename TValue> class SpMatrix;
	template<typename TValue> class SpVector;


	template <typename TNode, typename TEdge> class Graph;
}
