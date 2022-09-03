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
	namespace _scp
	{
		template<typename TValue, uint64_t N, uint64_t... Shape> struct StaticTensorTypeEncapsulator { using Type = StaticTensor<TValue, Shape...>; };
		template<typename TValue, uint64_t N> struct StaticTensorTypeEncapsulator<TValue, N> { using Type = TValue; };
	}

	template<typename TValue, uint64_t... Shape>
	class StaticTensor : public DenseTensor<TValue>
	{
		public:

			using SubTensor = typename _scp::StaticTensorTypeEncapsulator<TValue, Shape...>::Type;

			constexpr StaticTensor();
			constexpr StaticTensor(const TValue& value);
			constexpr StaticTensor(const TValue* values);
			constexpr StaticTensor(const std::vector<TValue>& values);
			constexpr StaticTensor(const std::initializer_list<TValue>& values);
			constexpr StaticTensor(const TensorBase<TValue>& tensor);
			constexpr StaticTensor(const StaticTensor<TValue, Shape...>& tensor);
			constexpr StaticTensor(StaticTensor<TValue, Shape...>&& tensor);

			constexpr StaticTensor<TValue, Shape...>& operator=(const StaticTensor<TValue, Shape...>& tensor);
			constexpr StaticTensor<TValue, Shape...>& operator=(StaticTensor<TValue, Shape...>&& tensor);

			constexpr virtual StaticTensor<TValue, Shape...>* clone() const override final;

			constexpr SubTensor& operator[](uint64_t i);
			constexpr const SubTensor& operator[](uint64_t i) const;
			
			constexpr virtual bool operator==(const TensorBase<TValue>& tensor) const override final;

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

			constexpr virtual TValue* getData() override final;
			constexpr virtual const TValue* getData() const override final;

			constexpr ~StaticTensor() = default;

		protected:

			constexpr virtual void create(uint64_t order, const uint64_t* shape) override final;
			constexpr virtual void copyFrom(const TensorBase<TValue>& tensor) override;
			constexpr virtual void destroy() override final;

			using DenseTensor<TValue>::_zero;

			static constexpr uint64_t _order = sizeof...(Shape);
			static constexpr uint64_t _shape[_order] = { Shape... };
			static constexpr uint64_t _length = (Shape * ...);
			static constexpr uint64_t _offset = _length / _shape[0];

			TValue _values[_length];
	};

	template<typename TValue, uint64_t... Shape>
	TensorIterator<sizeof...(Shape)> begin(const StaticTensor<TValue, Shape...>& tensor);
	template<typename TValue, uint64_t... Shape>
	TensorIterator<sizeof...(Shape)> end(const StaticTensor<TValue, Shape...>& tensor);

	template<typename TValue, uint64_t... ShapeA, uint64_t... ShapeB>
	StaticTensor<TValue, ShapeA..., ShapeB...> tensorProduct(const StaticTensor<TValue, ShapeA...>& a, const StaticTensor<TValue, ShapeB...>& b);
	template<typename TValue, uint64_t... ShapeA, uint64_t N, uint64_t... ShapeB>
	StaticTensor<TValue, ShapeA..., ShapeB...> contractedTensorProduct(const StaticTensor<TValue, ShapeA..., N>& a, const StaticTensor<TValue, N, ShapeB...>& b);


	template<typename TValue, uint64_t NRow, uint64_t NCol>
	class StaticMatrix : public DenseMatrix<StaticTensor<TValue, NRow, NCol>>
	{
		public:

			constexpr StaticMatrix();
			constexpr StaticMatrix(const TValue& value);
			constexpr StaticMatrix(const TValue* values);
			constexpr StaticMatrix(const std::vector<TValue>& values);
			constexpr StaticMatrix(const std::initializer_list<TValue>& values);
			constexpr StaticMatrix(const TensorBase<TValue>& tensor);
			constexpr StaticMatrix(const StaticMatrix<TValue, NRow, NCol>& matrix) = default;
			constexpr StaticMatrix(StaticMatrix<TValue, NRow, NCol>&& matrix) = default;

			constexpr StaticMatrix<TValue, NRow, NCol>& operator=(const StaticMatrix<TValue, NRow, NCol>& matrix) = default;
			constexpr StaticMatrix<TValue, NRow, NCol>& operator=(StaticMatrix<TValue, NRow, NCol>&& matrix) = default;

			constexpr virtual ~StaticMatrix() = default;

		private:

			using DenseMatrix<StaticTensor<TValue, NRow, NCol>>::_zero;
	};

	template<typename TValue, uint64_t NRow, uint64_t NMiddle, uint64_t NCol>
	StaticMatrix<TValue, NRow, NCol> operator*(const StaticMatrix<TValue, NRow, NMiddle>& a, const StaticMatrix<TValue, NMiddle, NCol>& b);
	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticVector<TValue, NRow> operator*(const StaticMatrix<TValue, NRow, NCol>& matrix, const StaticVector<TValue, NCol>& vector);
	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticMatrix<TValue, NCol, NRow> transpose(const StaticMatrix<TValue, NRow, NCol>& matrix);


	template<typename TValue, uint64_t Size>
	class StaticVector : public DenseVector<StaticTensor<TValue, Size>>
	{
		public:

			constexpr StaticVector();
			constexpr StaticVector(const TValue& value);
			constexpr StaticVector(const TValue* values);
			constexpr StaticVector(const std::vector<TValue>& values);
			constexpr StaticVector(const std::initializer_list<TValue>& values);
			constexpr StaticVector(const TensorBase<TValue>& tensor);
			constexpr StaticVector(const StaticVector<TValue, Size>& vector) = default;
			constexpr StaticVector(StaticVector<TValue, Size>&& vector) = default;

			constexpr StaticVector<TValue, Size>& operator=(const StaticVector<TValue, Size>& vector) = default;
			constexpr StaticVector<TValue, Size>& operator=(StaticVector<TValue, Size>&& vector) = default;

			constexpr virtual ~StaticVector() = default;

		private:

			using DenseVector<StaticTensor<TValue, Size>>::_zero;
	};

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticVector<TValue, NCol> operator*(const StaticVector<TValue, NRow>& vector, const StaticMatrix<TValue, NRow, NCol>& matrix);
}

#include <SciPP/Core/Tensor/templates/StaticTensor.hpp>
