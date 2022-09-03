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
	template<typename TValue>
	class DenseTensor : public TensorBase<TValue>
	{
		public:

			DenseTensor(const DenseTensor<TValue>& tensor) = delete;
			DenseTensor(DenseTensor<TValue>&& tensor) = delete;

			DenseTensor<TValue>& operator=(const DenseTensor<TValue>& tensor) = delete;
			DenseTensor<TValue>& operator=(DenseTensor<TValue>&& tensor) = delete;

			constexpr virtual DenseTensor<TValue>* clone() const override = 0;

			constexpr virtual DenseTensor<TValue>& operator+=(const TensorBase<TValue>& tensor) override final;
			constexpr virtual DenseTensor<TValue>& operator-=(const TensorBase<TValue>& tensor) override final;
			constexpr virtual DenseTensor<TValue>& operator*=(const TValue& value) override final;
			constexpr virtual DenseTensor<TValue>& operator/=(const TValue& value) override final;

			constexpr virtual bool operator==(const TensorBase<TValue>& tensor) const override;

			constexpr virtual DenseTensor<TValue>& hadamardProduct(const TensorBase<TValue>& tensor) override final;
			constexpr virtual DenseTensor<TValue>& negate() override final;
			constexpr virtual TValue dotProduct(const TensorBase<TValue>& tensor) const override final;
			constexpr virtual void tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const override final;

			constexpr virtual DenseTensor<TValue>& fill(const TValue& value) override final;
			constexpr virtual DenseTensor<TValue>& apply(const std::function<TValue(const TValue&)>& function) override final;

			constexpr virtual TValue normSq() const override final;
			constexpr virtual const TValue& minElement() const override final;
			constexpr virtual const TValue& maxElement() const override final;

			constexpr virtual const TValue& get(uint64_t internalIndex) const override = 0;
			constexpr virtual const TValue& get(const uint64_t* indices) const override = 0;
			constexpr virtual const TValue& get(const std::initializer_list<uint64_t>& indices) const override = 0;
			constexpr virtual void set(uint64_t internalIndex, const TValue& value) override = 0;
			constexpr virtual void set(const uint64_t* indices, const TValue& value) override = 0;
			constexpr virtual void set(const std::initializer_list<uint64_t>& indices, const TValue& value) override = 0;

			constexpr virtual void getIndices(uint64_t internalIndex, uint64_t* indices) const override;
			constexpr virtual uint64_t getInternalIndex(const uint64_t* indices) const override;
			constexpr virtual uint64_t getInternalLength() const override;

			constexpr virtual uint64_t getOrder() const override = 0;
			constexpr virtual const uint64_t* getShape() const override = 0;
			constexpr virtual uint64_t getSize(uint64_t i) const override = 0;

			constexpr virtual TValue* getData() = 0;
			constexpr virtual const TValue* getData() const = 0;

			constexpr virtual ~DenseTensor() = default;

		protected:

			constexpr DenseTensor() = default;

			constexpr virtual void setInitialPosition(TensorIteratorBase* iterator, bool end) const override final;
			constexpr virtual void incrementPosition(TensorIteratorBase* iterator) const override final;

			constexpr virtual void create(uint64_t order, const uint64_t* shape) override = 0;
			constexpr virtual void copyFrom(const TensorBase<TValue>& tensor) override;
			constexpr virtual void destroy() override = 0;

			using TensorBase<TValue>::_zero;
	};

	
	template<DenseTensorConcept TTensor>
	class DenseMatrix : public MatrixBase<TTensor>
	{
		public:

			using ValueType = TTensor::ValueType;

			// TODO: Other overrides ?
			constexpr virtual void matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const override final;
			constexpr virtual void vectorProduct(const TensorBase<ValueType>& vector, TensorBase<ValueType>& result) const override final;

		protected:

			using TTensor::_zero;

			template<typename... Args> constexpr DenseMatrix(Args... args);
			constexpr DenseMatrix(const TensorBase<ValueType>& tensor);
	};

	
	template<DenseTensorConcept TTensor>
	class DenseVector : public VectorBase<TTensor>
	{
		public:

			using ValueType = TTensor::ValueType;

			// TODO: Other overrides ?
			constexpr virtual void matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const;

		protected:

			using TTensor::_zero;

			template<typename... Args> constexpr DenseVector(Args... args);
			constexpr DenseVector(const TensorBase<ValueType>& tensor);
	};
}

#include <SciPP/Core/Tensor/templates/DenseTensor.hpp>
