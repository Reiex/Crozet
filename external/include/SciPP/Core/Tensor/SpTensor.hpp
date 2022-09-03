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
	template<typename TValue>
	class SpTensor : public TensorBase<TValue>
	{
		public:

			constexpr SpTensor(uint64_t order, const uint64_t* shape);
			constexpr SpTensor(uint64_t order, const uint64_t* shape, const TValue& value);
			constexpr SpTensor(uint64_t order, const uint64_t* shape, const TValue* values);
			constexpr SpTensor(uint64_t order, const uint64_t* shape, const std::vector<TValue>& values);
			constexpr SpTensor(uint64_t order, const uint64_t* shape, const std::initializer_list<TValue>& values);
			constexpr SpTensor(const std::vector<uint64_t>& shape);
			constexpr SpTensor(const std::vector<uint64_t>& shape, const TValue& value);
			constexpr SpTensor(const std::vector<uint64_t>& shape, const TValue* values);
			constexpr SpTensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr SpTensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr SpTensor(const std::initializer_list<uint64_t>& shape);
			constexpr SpTensor(const std::initializer_list<uint64_t>& shape, const TValue& value);
			constexpr SpTensor(const std::initializer_list<uint64_t>& shape, const TValue* values);
			constexpr SpTensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr SpTensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr SpTensor(const TensorBase<TValue>& tensor);
			constexpr SpTensor(const SpTensor<TValue>& tensor);
			constexpr SpTensor(SpTensor<TValue>&& tensor);

			constexpr SpTensor<TValue>& operator=(const SpTensor<TValue>& tensor);
			constexpr SpTensor<TValue>& operator=(SpTensor<TValue>&& tensor);

			constexpr virtual SpTensor<TValue>* clone() const override final;

			constexpr virtual SpTensor<TValue>& operator+=(const TensorBase<TValue>& tensor) override final;
			constexpr virtual SpTensor<TValue>& operator-=(const TensorBase<TValue>& tensor) override final;
			constexpr virtual SpTensor<TValue>& operator*=(const TValue& value) override final;
			constexpr virtual SpTensor<TValue>& operator/=(const TValue& value) override final;

			constexpr virtual bool operator==(const TensorBase<TValue>& tensor) const override final;

			constexpr virtual SpTensor<TValue>& hadamardProduct(const TensorBase<TValue>& tensor) override final;
			constexpr virtual SpTensor<TValue>& convolution(const TensorBase<TValue>& kernel, ConvolutionMethod method) override final;
			constexpr virtual SpTensor<TValue>& negate() override final;
			constexpr virtual TValue dotProduct(const TensorBase<TValue>& tensor) const override final;
			constexpr virtual void tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const override final;

			constexpr virtual TValue normSq() const override final;
			constexpr virtual const TValue& minElement() const override final;
			constexpr virtual const TValue& maxElement() const override final;

			constexpr virtual const TValue& get(uint64_t internalIndex) const override final;
			constexpr virtual const TValue& get(const uint64_t* indices) const override final;
			constexpr virtual const TValue& get(const std::initializer_list<uint64_t>& indices) const override final;
			constexpr virtual void set(uint64_t internalIndex, const TValue& value) override final;
			constexpr virtual void set(const uint64_t* indices, const TValue& value) override final;
			constexpr virtual void set(const std::initializer_list<uint64_t>& indices, const TValue& value) override final;

			constexpr virtual void getIndices(uint64_t internalIndex, uint64_t* indices) const override final;
			constexpr virtual uint64_t getInternalIndex(const uint64_t* indices) const override final;
			constexpr virtual uint64_t getInternalLength() const override final;

			constexpr virtual uint64_t getOrder() const override final;
			constexpr virtual const uint64_t* getShape() const override final;
			constexpr virtual uint64_t getSize(uint64_t i) const override final;

			constexpr ~SpTensor();

		protected:

			constexpr SpTensor();

			struct SpTensorElement
			{
				constexpr SpTensorElement(uint64_t eltOrder, const uint64_t* eltIndices, const TValue& eltValue);
				constexpr SpTensorElement(const SpTensorElement& elt);
				constexpr SpTensorElement(SpTensorElement&& elt);

				constexpr SpTensorElement& operator=(const SpTensorElement& elt);
				constexpr SpTensorElement& operator=(SpTensorElement&& elt);

				uint64_t order;
				uint64_t* indices;
				TValue value;

				constexpr ~SpTensorElement();
			};

			constexpr uint64_t findSpElement(const uint64_t* indices, bool& found) const;

			constexpr virtual void setInitialPosition(TensorIteratorBase* iterator, bool end) const override final;
			constexpr virtual void incrementPosition(TensorIteratorBase* iterator) const override final;

			constexpr virtual void create(uint64_t order, const uint64_t* shape) override final;
			constexpr virtual void copyFrom(const TensorBase<TValue>& tensor) override final;
			constexpr virtual void moveFrom(TensorBase<TValue>&& tensor) override final;
			constexpr virtual void destroy() override final;

			using TensorBase<TValue>::_zero;

			uint64_t _order;
			uint64_t* _shape;
			std::vector<SpTensorElement> _values;
	};

	template<typename TValue>
	constexpr DynTensorIterator begin(const SpTensor<TValue>& tensor);
	template<typename TValue>
	constexpr DynTensorIterator end(const SpTensor<TValue>& tensor);

	template<typename TValue>
	SpTensor<TValue> tensorContraction(const SpTensor<TValue>& tensor, uint64_t i, uint64_t j);
	template<typename TValue>
	SpTensor<TValue> tensorProduct(const SpTensor<TValue>& a, const SpTensor<TValue>& b);
	template<typename TValue>
	SpTensor<TValue> contractedTensorProduct(const SpTensor<TValue>& a, const SpTensor<TValue>& b);


	template<typename TValue>
	class SpMatrix : public MatrixBase<SpTensor<TValue>>
	{
		public:

			SpMatrix() = delete;
			constexpr SpMatrix(uint64_t row, uint64_t col);
			constexpr SpMatrix(uint64_t row, uint64_t col, const TValue& value);
			constexpr SpMatrix(uint64_t row, uint64_t col, const TValue* values);
			constexpr SpMatrix(uint64_t row, uint64_t col, const std::vector<TValue>& values);
			constexpr SpMatrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values);
			constexpr SpMatrix(const TensorBase<TValue>& tensor);
			constexpr SpMatrix(const SpMatrix<TValue>& matrix) = default;
			constexpr SpMatrix(SpMatrix<TValue>&& matrix) = default;

			constexpr SpMatrix<TValue>& operator=(const SpMatrix<TValue>& matrix) = default;
			constexpr SpMatrix<TValue>& operator=(SpMatrix<TValue>&& matrix) = default;

			// TODO: Overrides

			constexpr virtual ~SpMatrix() = default;

		private:

			using MatrixBase<SpTensor<TValue>>::_zero;
	};

	template<typename TValue>
	SpMatrix<TValue> operator*(const SpMatrix<TValue>& a, const SpMatrix<TValue>& b);
	template<typename TValue>
	SpVector<TValue> operator*(const SpMatrix<TValue>& matrix, const SpVector<TValue>& vector);
	template<typename TValue>
	SpMatrix<TValue> transpose(const SpMatrix<TValue>& matrix);


	template<typename TValue>
	class SpVector : public VectorBase<SpTensor<TValue>>
	{
		public:

			SpVector() = delete;
			constexpr SpVector(uint64_t size);
			constexpr SpVector(uint64_t size, const TValue& value);
			constexpr SpVector(uint64_t size, const TValue* values);
			constexpr SpVector(const std::vector<TValue>& values);
			constexpr SpVector(const std::initializer_list<TValue>& values);
			constexpr SpVector(const TensorBase<TValue>& tensor);
			constexpr SpVector(const SpVector<TValue>& vector) = default;
			constexpr SpVector(SpVector<TValue>&& vector) = default;

			constexpr SpVector<TValue>& operator=(const SpVector<TValue>& vector) = default;
			constexpr SpVector<TValue>& operator=(SpVector<TValue>&& vector) = default;

			// TODO: Overrides

			constexpr virtual ~SpVector() = default;

		private:

			using VectorBase<SpTensor<TValue>>::_zero;
	};

	template<typename TValue>
	SpVector<TValue> operator*(const SpVector<TValue>& vector, const SpMatrix<TValue>& matrix);
}

#include <SciPP/Core/Tensor/templates/SpTensor.hpp>
