#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/TensorBase.hpp>

namespace scp
{
	template<TensorConcept TTensor>
	class MatrixBase : public TTensor
	{
		public:

			using TensorType = TTensor;
			using ValueType = TTensor::ValueType;

			constexpr virtual TensorBase<ValueType>& inverse();
			constexpr virtual void matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const;
			constexpr virtual void vectorProduct(const TensorBase<ValueType>& vector, TensorBase<ValueType>& result) const;
			constexpr virtual void transpose(TensorBase<ValueType>& result) const;

			constexpr virtual ValueType determinant() const;

			constexpr virtual ~MatrixBase() = default;

		protected:

			using TTensor::_zero;

			template<typename... Args> constexpr MatrixBase(Args... args);
			constexpr MatrixBase(const TensorBase<ValueType>& tensor);
	};

	template<TensorConcept TTensor>
	TensorIterator<2> begin(const MatrixBase<TTensor>& matrix);
	template<TensorConcept TTensor>
	TensorIterator<2> end(const MatrixBase<TTensor>& matrix);

	template<MatrixConcept TMatrix>
	constexpr TMatrix inverse(const TMatrix& matrix);


	template<TensorConcept TTensor>
	class VectorBase : public TTensor
	{
		public:

			using TensorType = TTensor;
			using ValueType = TTensor::ValueType;

			constexpr virtual TensorBase<ValueType>& crossProduct(const TensorBase<ValueType>& vector);
			constexpr virtual void matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const;

			constexpr virtual ~VectorBase() = default;

		protected:

			using TTensor::_zero;

			template<typename... Args> constexpr VectorBase(Args... args);
			constexpr VectorBase(const TensorBase<ValueType>& tensor);
	};

	template<TensorConcept TTensor>
	TensorIterator<1> begin(const VectorBase<TTensor>& vector);
	template<TensorConcept TTensor>
	TensorIterator<1> end(const VectorBase<TTensor>& vector);

	template<VectorConcept TVector>
	constexpr TVector crossProduct(const TVector& a, const TVector& b);
}

#include <SciPP/Core/Tensor/templates/MatrixBase.hpp>
