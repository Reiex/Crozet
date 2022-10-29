///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/TensorBase.hpp>

namespace scp
{
	template<typename TValue>
	class Tensor
	{
		SCP_TENSOR_DECL(Tensor, Tensor<TValue>)

		public:

			// Tensors API

			constexpr Tensor<TValue>& fill(const TValue& value);
			template<std::input_iterator TInput> constexpr Tensor<TValue>& copy(TInput it);
			constexpr Tensor<TValue>& transform(const std::function<TValue(const TValue&)>& unaryOp);
			template<std::input_iterator TInput> constexpr Tensor<TValue>& transform(TInput it, const std::function<TValue(const TValue&, const typename std::iterator_traits<TInput>::value_type&)>& binaryOp);

			template<TensorConcept<TValue> TTensor> constexpr Tensor<TValue>& operator+=(const TTensor& tensor);
			template<TensorConcept<TValue> TTensor> constexpr Tensor<TValue>& operator-=(const TTensor& tensor);
			template<typename TScalar> constexpr Tensor<TValue>& operator*=(const TScalar& scalar);
			template<typename TScalar> constexpr Tensor<TValue>& operator/=(const TScalar& scalar);
			constexpr Tensor<TValue>& negate();

			template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB> constexpr void tensorProduct(const TTensorA& tensorA, const TTensorB& tensorB);
			template<TensorConcept<TValue> TTensor> constexpr Tensor<TValue>& hadamardProduct(const TTensor& tensor);
			constexpr Tensor<TValue>& fft();
			constexpr Tensor<TValue>& ifft();

			template<BorderBehaviour BBehaviour, TensorConcept<TValue> TTensor> constexpr Tensor<TValue>& convolution(const TTensor& kernel);

			template<typename TScalar, InterpolationMethod IMethod, TensorConcept<TValue> TTensor> constexpr void interpolation(const TTensor& tensor);
			template<TensorConcept<TValue> TTensor> constexpr void tensorContraction(const TTensor& tensor, uint64_t i, uint64_t j);

			template<TensorConcept<TValue> TTensor> constexpr bool operator==(const TTensor& tensor) const;
			template<TensorConcept<TValue> TTensor> constexpr bool operator!=(const TTensor& tensor) const;

			template<TensorConcept<TValue> TTensor> constexpr TValue innerProduct(const TTensor& tensor) const;

			constexpr TValue norm() const;
			constexpr const TValue& minElement(const std::function<bool(const TValue&, const TValue&)>& compare = std::less<TValue>()) const;
			constexpr const TValue& maxElement(const std::function<bool(const TValue&, const TValue&)>& compare = std::less<TValue>()) const;
			template<typename TScalar, InterpolationMethod IMethod> constexpr TValue getInterpolated(const TScalar* scalarIndices) const;
			template<typename TScalar, InterpolationMethod IMethod> constexpr TValue getInterpolated(const std::initializer_list<TScalar>& scalarIndices) const;
			template<BorderBehaviour BBehaviour> constexpr const TValue& getOutOfBound(const int64_t* indices) const;
			template<BorderBehaviour BBehaviour> constexpr const TValue& getOutOfBound(const std::initializer_list<int64_t>& indices) const;


			constexpr TValue* begin();
			constexpr TValue* end();
			constexpr const TValue* begin() const;
			constexpr const TValue* end() const;
			constexpr const TValue* cbegin() const;
			constexpr const TValue* cend() const;

			constexpr const TValue& get(uint64_t index) const;
			constexpr const TValue& get(const uint64_t* indices) const;
			constexpr const TValue& get(const std::initializer_list<uint64_t>& indices) const;
			constexpr void set(uint64_t index, const TValue& value);
			constexpr void set(const uint64_t* indices, const TValue& value);
			constexpr void set(const std::initializer_list<uint64_t>& indices, const TValue& value);

			constexpr uint64_t getOrder() const;
			constexpr const uint64_t* getSizes() const;
			constexpr uint64_t getSize(uint64_t i) const;
			constexpr uint64_t getElementCount() const;

			// Out of API

			static constexpr Tensor<TValue>* createAroundMemory(uint64_t order, const uint64_t* sizes, TValue* memory);
			static constexpr Tensor<TValue>* createAroundMemory(const std::initializer_list<uint64_t>& sizes, TValue* memory);


			constexpr TValue& operator[](uint64_t index);
			constexpr TValue& operator[](const std::initializer_list<uint64_t>& indices);
			constexpr const TValue& operator[](uint64_t index) const;
			constexpr const TValue& operator[](const std::initializer_list<uint64_t>& indices) const;

			constexpr TValue& get(uint64_t index);
			constexpr TValue& get(const uint64_t* indices);
			constexpr TValue& get(const std::initializer_list<uint64_t>& indices);

			constexpr TValue* getData();
			constexpr const TValue* getData() const;


			constexpr virtual ~Tensor();

		protected:

			constexpr Tensor();

			constexpr void _create(uint64_t order, const uint64_t* sizes);
			template<TensorConcept<TValue> TTensor> constexpr void _copyFrom(const TTensor& tensor);
			template<TensorConcept<TValue> TTensor> constexpr void _moveFrom(TTensor&& tensor);
			constexpr void _destroy();

			constexpr void _cooleyTukey(const TValue* const* exponentials, const uint64_t* sizes, uint64_t* offset, uint64_t* stride);

			uint64_t _order;
			uint64_t _length;
			uint64_t* _sizes;
			TValue* _values;

			bool _owner;
	};

	template<typename TValue>
	class Matrix : public Tensor<TValue>
	{
		SCP_MATRIX_DECL(Matrix, Matrix<TValue>, Tensor<TValue>)
	
		public:

			// Tensors API
			
			template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB> constexpr void matrixProduct(const TTensorA& matrixA, const TTensorB& matrixB);
			
			constexpr Matrix<TValue>& transpose();
			
			constexpr Matrix<TValue>& inverse();  
			constexpr TValue determinant() const;

			// Out of API

			static constexpr Matrix<TValue>* createAroundMemory(uint64_t row, uint64_t col, TValue* memory);

			constexpr ~Matrix() = default;

		private:

			constexpr Matrix() = default;

			using Tensor<TValue>::_order;
			using Tensor<TValue>::_length;
			using Tensor<TValue>::_sizes;
			using Tensor<TValue>::_values;
			using Tensor<TValue>::_owner;

		friend class Tensor<TValue>;
	};
	
	
	template<typename TValue>
	class Vector : public Tensor<TValue>
	{
		SCP_VECTOR_DECL(Vector, Vector<TValue>, Tensor<TValue>)
	
		public:

			// Tensors API
	
			template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB> constexpr void rightMatrixProduct(const TTensorA& vector, const TTensorB& matrix) const;
			template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB> constexpr void leftMatrixProduct(const TTensorA& matrix, const TTensorB& vector) const;
			template<TensorConcept<TValue> TTensor> constexpr Vector<TValue>& crossProduct(const TTensor& vector);

			// Out of API
	
			static constexpr Vector<TValue>* createAroundMemory(uint64_t size, TValue* memory);

			constexpr ~Vector() = default;

		private:

			constexpr Vector() = default;

			using Tensor<TValue>::_order;
			using Tensor<TValue>::_length;
			using Tensor<TValue>::_sizes;
			using Tensor<TValue>::_values;
			using Tensor<TValue>::_owner;

		friend class Tensor<TValue>;
	};
}

#include <SciPP/Core/Tensor/templates/Tensor.hpp>
