#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/TensorBase.hpp>
#include <SciPP/Core/Tensor/MatrixBase.hpp>

namespace scp
{
	template <typename TValue>
	class DynTensor : public DenseTensor<TValue>
	{
		public:

			constexpr DynTensor(const uint64_t order, const uint64_t* shape);
			constexpr DynTensor(const uint64_t order, const uint64_t* shape, const TValue& value);
			constexpr DynTensor(const uint64_t order, const uint64_t* shape, const TValue* values);
			constexpr DynTensor(const uint64_t order, const uint64_t* shape, const std::vector<TValue>& values);
			constexpr DynTensor(const uint64_t order, const uint64_t* shape, const std::initializer_list<TValue>& values);
			constexpr DynTensor(const std::vector<uint64_t>& shape);
			constexpr DynTensor(const std::vector<uint64_t>& shape, const TValue& value);
			constexpr DynTensor(const std::vector<uint64_t>& shape, const TValue* values);
			constexpr DynTensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr DynTensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr DynTensor(const std::initializer_list<uint64_t>& shape);
			constexpr DynTensor(const std::initializer_list<uint64_t>& shape, const TValue& value);
			constexpr DynTensor(const std::initializer_list<uint64_t>& shape, const TValue* values);
			constexpr DynTensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr DynTensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr DynTensor(const TensorBase<TValue>& tensor);
			constexpr DynTensor(const DynTensor<TValue>& tensor);
			constexpr DynTensor(DynTensor<TValue>&& tensor);

			constexpr DynTensor<TValue>& operator=(const DynTensor<TValue>& tensor);
			constexpr DynTensor<TValue>& operator=(DynTensor<TValue>&& tensor);

			constexpr virtual DynTensor<TValue>* clone() const override final;

			constexpr virtual const TValue& get(uint64_t internalIndex) const override final;
			constexpr virtual const TValue& get(const uint64_t* indices) const override final;
			constexpr virtual const TValue& get(const std::initializer_list<uint64_t>& indices) const override final;
			constexpr virtual void set(uint64_t internalIndex, const TValue& value) override final;
			constexpr virtual void set(const uint64_t* indices, const TValue& value) override final;
			constexpr virtual void set(const std::initializer_list<uint64_t>& indices, const TValue& value) override final;

			constexpr virtual uint64_t getInternalLength() const override final;

			constexpr virtual uint64_t getOrder() const override final;
			constexpr virtual const uint64_t* getShape() const override final;
			constexpr virtual uint64_t getSize(uint64_t i) const override final;
			constexpr virtual uint64_t getTotalLength() const override final;

			constexpr virtual TValue* getData() override final;
			constexpr virtual const TValue* getData() const override final;

			constexpr virtual ~DynTensor();

		protected:

			constexpr DynTensor();

			constexpr virtual void create(uint64_t order, const uint64_t* shape) override final;
			constexpr virtual void moveFrom(TensorBase<TValue>&& tensor) override final;
			constexpr virtual void destroy() override final;

			using DenseTensor<TValue>::_zero;

			TValue* _values;
			uint64_t _order;
			uint64_t* _shape;
			uint64_t _length;
	};

	template<typename TValue>
	constexpr DynTensorIterator begin(const DynTensor<TValue>& tensor);
	template<typename TValue>
	constexpr DynTensorIterator end(const DynTensor<TValue>& tensor);

	template<typename TValue>
	DynTensor<TValue> tensorContraction(const DynTensor<TValue>& tensor, uint64_t i, uint64_t j);
	template<typename TValue>
	DynTensor<TValue> tensorProduct(const DynTensor<TValue>& a, const DynTensor<TValue>& b);
	template<typename TValue>
	DynTensor<TValue> contractedTensorProduct(const DynTensor<TValue>& a, const DynTensor<TValue>& b);


	template<typename TValue>
	class DynMatrix : public DenseMatrix<DynTensor<TValue>>
	{
		public:

			DynMatrix() = delete;
			constexpr DynMatrix(uint64_t row, uint64_t col);
			constexpr DynMatrix(uint64_t row, uint64_t col, const TValue& value);
			constexpr DynMatrix(uint64_t row, uint64_t col, const TValue* values);
			constexpr DynMatrix(uint64_t row, uint64_t col, const std::vector<TValue>& values);
			constexpr DynMatrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values);
			constexpr DynMatrix(const TensorBase<TValue>& tensor);
			constexpr DynMatrix(const DynMatrix<TValue>& matrix) = default;
			constexpr DynMatrix(DynMatrix<TValue>&& matrix) = default;

			constexpr DynMatrix<TValue>& operator=(const DynMatrix<TValue>& matrix) = default;
			constexpr DynMatrix<TValue>& operator=(DynMatrix<TValue>&& matrix) = default;

			constexpr virtual ~DynMatrix() = default;

		private:

			using DenseMatrix<DynTensor<TValue>>::_zero;
	};

	template<typename TValue>
	DynMatrix<TValue> operator*(const DynMatrix<TValue>& a, const DynMatrix<TValue>& b);
	template<typename TValue>
	DynVector<TValue> operator*(const DynMatrix<TValue>& matrix, const DynVector<TValue>& vector);
	template<typename TValue>
	DynMatrix<TValue> transpose(const DynMatrix<TValue>& matrix);


	template<typename TValue>
	class DynVector : public DenseVector<DynTensor<TValue>>
	{
		public:

			DynVector() = delete;
			constexpr DynVector(uint64_t size);
			constexpr DynVector(uint64_t size, const TValue& value);
			constexpr DynVector(uint64_t size, const TValue* values);
			constexpr DynVector(const std::vector<TValue>& values);
			constexpr DynVector(const std::initializer_list<TValue>& values);
			constexpr DynVector(const TensorBase<TValue>& tensor);
			constexpr DynVector(const DynVector<TValue>& vector) = default;
			constexpr DynVector(DynVector<TValue>&& vector) = default;

			constexpr DynVector<TValue>& operator=(const DynVector<TValue>& vector) = default;
			constexpr DynVector<TValue>& operator=(DynVector<TValue>&& vector) = default;

			constexpr virtual ~DynVector() = default;

		private:

			using DenseVector<DynTensor<TValue>>::_zero;
	};

	template<typename TValue>
	DynVector<TValue> operator*(const DynVector<TValue>& vector, const DynMatrix<TValue>& matrix);
}

#include <SciPP/Core/Tensor/templates/DynTensor.hpp>
