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
		template<typename TValue, uint64_t Order> struct TensorTypeEncapsulator { using Type = Tensor<TValue, Order>; };
		template<typename TValue> struct TensorTypeEncapsulator<TValue, 0> { using Type = TValue; };
	}

	template <typename TValue, uint64_t Order>
	class Tensor : public DenseTensor<TValue>
	{
		public:

			using SubTensor = typename _scp::TensorTypeEncapsulator<TValue, Order - 1>::Type;

			constexpr Tensor(const uint64_t* shape);
			constexpr Tensor(const uint64_t* shape, const TValue& value);
			constexpr Tensor(const uint64_t* shape, const TValue* values);
			constexpr Tensor(const uint64_t* shape, const std::vector<TValue>& values);
			constexpr Tensor(const uint64_t* shape, const std::initializer_list<TValue>& values);
			constexpr Tensor(const std::vector<uint64_t>& shape);
			constexpr Tensor(const std::vector<uint64_t>& shape, const TValue& value);
			constexpr Tensor(const std::vector<uint64_t>& shape, const TValue* values);
			constexpr Tensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr Tensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr Tensor(const std::initializer_list<uint64_t>& shape);
			constexpr Tensor(const std::initializer_list<uint64_t>& shape, const TValue& value);
			constexpr Tensor(const std::initializer_list<uint64_t>& shape, const TValue* values);
			constexpr Tensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values);
			constexpr Tensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values);
			constexpr Tensor(const TensorBase<TValue>& tensor);
			constexpr Tensor(const Tensor<TValue, Order>& tensor);
			constexpr Tensor(Tensor<TValue, Order>&& tensor);

			constexpr Tensor<TValue, Order>& operator=(const Tensor<TValue, Order>& tensor);
			constexpr Tensor<TValue, Order>& operator=(Tensor<TValue, Order>&& tensor);

			constexpr virtual Tensor<TValue, Order>* clone() const override final;

			constexpr SubTensor& operator[](uint64_t i);
			constexpr const SubTensor& operator[](uint64_t i) const;

			constexpr virtual Tensor<TValue, Order>& fft() override final;
			constexpr virtual Tensor<TValue, Order>& ifft() override final;

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
			constexpr virtual uint64_t getTotalLength() const override final;

			constexpr virtual TValue* getData() override final;
			constexpr virtual const TValue* getData() const override final;

			constexpr virtual ~Tensor();

		protected:

			constexpr Tensor();

			constexpr void initSubTensor(TValue* values, uint64_t* shape, uint64_t length, uint64_t treeLength);

			constexpr virtual void create(uint64_t order, const uint64_t* shape) override final;
			constexpr virtual void copyFrom(const TensorBase<TValue>& tensor) override final;
			constexpr virtual void moveFrom(TensorBase<TValue>&& tensor) override final;
			constexpr virtual void destroy() override final;

			using DenseTensor<TValue>::_zero;

			TValue* _values;
			SubTensor* _tree;
			uint64_t* _shape;
			uint64_t _treeOffset;
			uint64_t _length;
			bool _owner;

		friend class Tensor<TValue, Order + 1>;
	};

	template<typename TValue, uint64_t Order>
	constexpr TensorIterator<Order> begin(const Tensor<TValue, Order>& tensor);
	template<typename TValue, uint64_t Order>
	constexpr TensorIterator<Order> end(const Tensor<TValue, Order>& tensor);

	template<typename TValue, uint64_t Order>
	Tensor<TValue, Order - 2> tensorContraction(const Tensor<TValue, Order>& tensor, uint64_t i, uint64_t j);
	template<typename TValue, uint64_t OrderA, uint64_t OrderB>
	Tensor<TValue, OrderA + OrderB> tensorProduct(const Tensor<TValue, OrderA>& a, const Tensor<TValue, OrderB>& b);
	template<typename TValue, uint64_t OrderA, uint64_t OrderB>
	Tensor<TValue, OrderA + OrderB - 2> contractedTensorProduct(const Tensor<TValue, OrderA>& a, const Tensor<TValue, OrderB>& b);


	template<typename TValue>
	class Matrix : public DenseMatrix<Tensor<TValue, 2>>
	{
		public:

			Matrix() = delete;
			constexpr Matrix(uint64_t row, uint64_t col);
			constexpr Matrix(uint64_t row, uint64_t col, const TValue& value);
			constexpr Matrix(uint64_t row, uint64_t col, const TValue* values);
			constexpr Matrix(uint64_t row, uint64_t col, const std::vector<TValue>& values);
			constexpr Matrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values);
			constexpr Matrix(const TensorBase<TValue>& tensor);
			constexpr Matrix(const Matrix<TValue>& matrix) = default;
			constexpr Matrix(Matrix<TValue>&& matrix) = default;

			constexpr Matrix<TValue>& operator=(const Matrix<TValue>& matrix) = default;
			constexpr Matrix<TValue>& operator=(Matrix<TValue>&& matrix) = default;

			constexpr Vector<TValue>& operator[](uint64_t i);
			constexpr const Vector<TValue>& operator[](uint64_t i) const;

			constexpr virtual ~Matrix() = default;

		private:

			using DenseMatrix<Tensor<TValue, 2>>::_zero;
	};

	template<typename TValue>
	Matrix<TValue> operator*(const Matrix<TValue>& a, const Matrix<TValue>& b);
	template<typename TValue>
	Vector<TValue> operator*(const Matrix<TValue>& matrix, const Vector<TValue>& vector);
	template<typename TValue>
	Matrix<TValue> transpose(const Matrix<TValue>& matrix);


	template<typename TValue>
	class Vector : public DenseVector<Tensor<TValue, 1>>
	{
		public:

			Vector() = delete;
			constexpr Vector(uint64_t size);
			constexpr Vector(uint64_t size, const TValue& value);
			constexpr Vector(uint64_t size, const TValue* values);
			constexpr Vector(const std::vector<TValue>& values);
			constexpr Vector(const std::initializer_list<TValue>& values);
			constexpr Vector(const TensorBase<TValue>& tensor);
			constexpr Vector(const Vector<TValue>& vector) = default;
			constexpr Vector(Vector<TValue>&& vector) = default;

			constexpr Vector<TValue>& operator=(const Vector<TValue>& vector) = default;
			constexpr Vector<TValue>& operator=(Vector<TValue>&& vector) = default;

			constexpr virtual ~Vector() = default;

		private:

			using DenseVector<Tensor<TValue, 1>>::_zero;
	};

	template<typename TValue>
	Vector<TValue> operator*(const Vector<TValue>& vector, const Matrix<TValue>& matrix);
}

#include <SciPP/Core/Tensor/templates/Tensor.hpp>
