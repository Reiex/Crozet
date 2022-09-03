#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/DynTensor.hpp>

namespace scp
{
	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor() :
		_values(nullptr),
		_order(0),
		_shape(nullptr),
		_length(0)
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const uint64_t order, const uint64_t* shape) : DynTensor<TValue>()
	{
		create(order, shape);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const uint64_t order, const uint64_t* shape, const TValue& value) : DynTensor<TValue>(order, shape)
	{
		std::fill(_values, _values + _length, value);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const uint64_t order, const uint64_t* shape, const TValue* values) : DynTensor<TValue>(order, shape)
	{
		std::copy(values, values + _length, _values);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const uint64_t order, const uint64_t* shape, const std::vector<TValue>& values) : DynTensor<TValue>(order, shape, values.data())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const uint64_t order, const uint64_t* shape, const std::initializer_list<TValue>& values) : DynTensor<TValue>(order, shape, values.begin())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::vector<uint64_t>& shape) : DynTensor<TValue>(shape.size(), shape.data())
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::vector<uint64_t>& shape, const TValue& value) : DynTensor<TValue>(shape.size(), shape.data(), value)
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::vector<uint64_t>& shape, const TValue* values) : DynTensor<TValue>(shape.size(), shape.data(), values)
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values) : DynTensor<TValue>(shape.size(), shape.data(), values.data())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values) : DynTensor<TValue>(shape.size(), shape.data(), values.begin())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::initializer_list<uint64_t>& shape) : DynTensor<TValue>(shape.size(), shape.begin())
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::initializer_list<uint64_t>& shape, const TValue& value) : DynTensor<TValue>(shape.size(), shape.begin(), value)
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::initializer_list<uint64_t>& shape, const TValue* values) : DynTensor<TValue>(shape.size(), shape.begin(), values)
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values) : DynTensor<TValue>(shape.size(), shape.begin(), values.data())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values) : DynTensor<TValue>(shape.size(), shape.begin(), values.begin())
	{
		assert(values.size() == _length);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const TensorBase<TValue>& tensor) : DynTensor<TValue>()
	{
		create(tensor.getOrder(), tensor.getShape());
		this->copyFrom(tensor);
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(const DynTensor<TValue>& tensor) : DynTensor<TValue>(dynamic_cast<const TensorBase<TValue>&>(tensor))
	{
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::DynTensor(DynTensor<TValue>&& tensor) : DynTensor<TValue>()
	{
		moveFrom(std::move(tensor));
	}

	template<typename TValue>
	constexpr DynTensor<TValue>& DynTensor<TValue>::operator=(const DynTensor<TValue>& tensor)
	{
		copyFrom(tensor);
		return *this;
	}

	template<typename TValue>
	constexpr DynTensor<TValue>& DynTensor<TValue>::operator=(DynTensor<TValue>&& tensor)
	{
		moveFrom(std::move(tensor));
		return *this;
	}

	template<typename TValue>
	constexpr DynTensor<TValue>* DynTensor<TValue>::clone() const
	{
		return new DynTensor<TValue>(*this);
	}

	template<typename TValue>
	constexpr const TValue& DynTensor<TValue>::get(uint64_t internalIndex) const
	{
		assert(internalIndex < _length);
		return _values[internalIndex];
	}

	template<typename TValue>
	constexpr const TValue& DynTensor<TValue>::get(const uint64_t* indices) const
	{
		return get(this->getInternalIndex(indices));
	}

	template<typename TValue>
	constexpr const TValue& DynTensor<TValue>::get(const std::initializer_list<uint64_t>& indices) const
	{
		assert(indices.size() < _order);
		return get(this->getInternalIndex(indices.begin()));
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::set(uint64_t internalIndex, const TValue& value)
	{
		assert(internalIndex < _length);
		_values[internalIndex] = value;
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::set(const uint64_t* indices, const TValue& value)
	{
		set(this->getInternalIndex(indices), value);
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::set(const std::initializer_list<uint64_t>& indices, const TValue& value)
	{
		assert(indices.size() < _order);
		set(this->getInternalIndex(indices.begin()), value);
	}

	template<typename TValue>
	constexpr uint64_t DynTensor<TValue>::getInternalLength() const
	{
		return _length;
	}

	template<typename TValue>
	constexpr uint64_t DynTensor<TValue>::getOrder() const
	{
		return _order;
	}

	template<typename TValue>
	constexpr const uint64_t* DynTensor<TValue>::getShape() const
	{
		return _shape;
	}

	template<typename TValue>
	constexpr uint64_t DynTensor<TValue>::getSize(uint64_t i) const
	{
		assert(i < _order);
		return _shape[i];
	}

	template<typename TValue>
	constexpr uint64_t DynTensor<TValue>::getTotalLength() const
	{
		return _length;
	}

	template<typename TValue>
	constexpr TValue* DynTensor<TValue>::getData()
	{
		return _values;
	}

	template<typename TValue>
	constexpr const TValue* DynTensor<TValue>::getData() const
	{
		return _values;
	}

	template<typename TValue>
	constexpr DynTensor<TValue>::~DynTensor()
	{
		destroy();
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::create(uint64_t order, const uint64_t* shape)
	{
		assert(order > 0);
		assert(_values == nullptr);

		_order = order;

		_shape = new uint64_t[_order];
		std::copy(shape, shape + _order, _shape);

		_length = std::accumulate(_shape, _shape + _order, 1, std::multiplies<uint64_t>());
		assert(_length != 0);

		_values = new TValue[_length];
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::moveFrom(TensorBase<TValue>&& tensor)
	{
		DynTensor<TValue>* dynTensor = dynamic_cast<DynTensor<TValue>*>(&tensor);
		if (dynTensor)
		{
			destroy();

			_values = dynTensor->_values;
			_order = dynTensor->_order;
			_shape = dynTensor->_shape;
			_length = dynTensor->_length;

			dynTensor->_values = nullptr;
			dynTensor->_order = 0;
			dynTensor->_shape = nullptr;
			dynTensor->_length = 0;
		}
		else
		{
			DenseTensor<TValue>::moveFrom(std::move(tensor));
		}
	}

	template<typename TValue>
	constexpr void DynTensor<TValue>::destroy()
	{
		if (_values)
		{
			delete[] _shape;
			delete[] _values;

			_order = 0;
			_shape = nullptr;
			_length = 0;
			_values = nullptr;
		}
	}


	template<typename TValue>
	constexpr DynTensorIterator begin(const DynTensor<TValue>& tensor)
	{
		return DynTensorIterator(&tensor, false);
	}

	template<typename TValue>
	constexpr DynTensorIterator end(const DynTensor<TValue>& tensor)
	{
		return DynTensorIterator(&tensor, true);
	}


	template<typename TValue>
	DynTensor<TValue> tensorContraction(const DynTensor<TValue>& tensor, uint64_t i, uint64_t j)
	{

	}

	template<typename TValue>
	DynTensor<TValue> tensorProduct(const DynTensor<TValue>& a, const DynTensor<TValue>& b)
	{

	}

	template<typename TValue>
	DynTensor<TValue> contractedTensorProduct(const DynTensor<TValue>& a, const DynTensor<TValue>& b)
	{

	}


	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(uint64_t row, uint64_t col) : DenseMatrix<DynTensor<TValue>>(std::vector<uint64_t>{ { row, col } })
	{
	}

	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(uint64_t row, uint64_t col, const TValue& value) : DenseMatrix<DynTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, value)
	{
	}

	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(uint64_t row, uint64_t col, const TValue* values) : DenseMatrix<DynTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(uint64_t row, uint64_t col, const std::vector<TValue>& values) : DenseMatrix<DynTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values) : DenseMatrix<DynTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr DynMatrix<TValue>::DynMatrix(const TensorBase<TValue>& tensor) : DenseMatrix<DynTensor<TValue>>(tensor)
	{
	}


	template<typename TValue>
	DynMatrix<TValue> operator*(const DynMatrix<TValue>& a, const DynMatrix<TValue>& b)
	{
		DynMatrix<TValue> result(a.getSize(0), b.getSize(1));
		a.matrixProduct(b, result);
		return result;
	}

	template<typename TValue>
	DynVector<TValue> operator*(const DynMatrix<TValue>& matrix, const DynVector<TValue>& vector)
	{
		DynVector<TValue> result(matrix.getSize(0));
		matrix.vectorProduct(vector, result);
		return result;
	}

	template<typename TValue>
	DynMatrix<TValue> transpose(const DynMatrix<TValue>& matrix)
	{
		DynMatrix<TValue> result(matrix.getSize(1), matrix.getSize(0));
		matrix.transpose(result);
		return result;
	}


	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(uint64_t size) : DenseVector<Tensor<TValue, 1>>(&size)
	{
	}

	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(uint64_t size, const TValue& value) : DenseVector<DynTensor<TValue>>(1, &size, value)
	{
	}

	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(uint64_t size, const TValue* values) : DenseVector<DynTensor<TValue>>(1, &size, values)
	{
	}

	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(const std::vector<TValue>& values) : DenseVector<DynTensor<TValue>>(std::vector<uint64_t>{ { values.size() } }, values.data())
	{
	}

	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(const std::initializer_list<TValue>& values) : DenseVector<DynTensor<TValue>>(std::vector<uint64_t>{ { values.size() } }, values.begin())
	{
	}

	template<typename TValue>
	constexpr DynVector<TValue>::DynVector(const TensorBase<TValue>& tensor) : DenseVector<DynTensor<TValue>>(tensor)
	{
	}


	template<typename TValue>
	DynVector<TValue> operator*(const DynVector<TValue>& vector, const DynMatrix<TValue>& matrix)
	{
		DynVector<TValue> result(matrix.getSize(1));
		vector.matrixProduct(matrix, result);
		return result;
	}
}
