#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/StaticTensor.hpp>

namespace scp
{
	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor()
	{
		static_assert(_order != 0);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const TValue& value) : StaticTensor<TValue, Shape...>()
	{
		std::fill(_values, _values + _length, value);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const TValue* values) : StaticTensor<TValue, Shape...>()
	{
		std::copy(values, values + _length, _values);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const std::vector<TValue>& values) : StaticTensor<TValue, Shape...>()
	{
		assert(values.size() == _length);
		std::copy(values.begin(), values.end(), _values);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const std::initializer_list<TValue>& values) : StaticTensor<TValue, Shape...>()
	{
		assert(values.size() == _length);
		std::copy(values.begin(), values.end(), _values);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const TensorBase<TValue>& tensor) : StaticTensor<TValue, Shape...>()
	{
		copyFrom(tensor);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(const StaticTensor<TValue, Shape...>& tensor) : StaticTensor<TValue, Shape...>(dynamic_cast<const TensorBase<TValue>&>(tensor))
	{
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::StaticTensor(StaticTensor<TValue, Shape...>&& tensor) : StaticTensor<TValue, Shape...>()
	{
		this->moveFrom(std::move(tensor));
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>& StaticTensor<TValue, Shape...>::operator=(const StaticTensor<TValue, Shape...>& tensor)
	{
		copyFrom(tensor);
		return *this;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>& StaticTensor<TValue, Shape...>::operator=(StaticTensor<TValue, Shape...>&& tensor)
	{
		this->moveFrom(std::move(tensor));
		return *this;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>* StaticTensor<TValue, Shape...>::clone() const
	{
		return new StaticTensor<TValue, Shape...>(*this);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr StaticTensor<TValue, Shape...>::SubTensor& StaticTensor<TValue, Shape...>::operator[](uint64_t i)
	{
		assert(i < _shape[0]);
		return static_cast<SubTensor&>(_values[i * _offset]);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const StaticTensor<TValue, Shape...>::SubTensor& StaticTensor<TValue, Shape...>::operator[](uint64_t i) const
	{
		assert(i < _shape[0]);
		return static_cast<SubTensor&>(_values[i * _offset]);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr bool StaticTensor<TValue, Shape...>::operator==(const TensorBase<TValue>& tensor) const
	{
		const StaticTensor<TValue, Shape...>* staticTensor = dynamic_cast<const StaticTensor<TValue, Shape...>*>(&tensor);
		if (staticTensor)
		{
			return std::equal(_values, _values + _length, staticTensor->_values);
		}
		else
		{
			return DenseTensor<TValue>::operator==(tensor);
		}
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const TValue& StaticTensor<TValue, Shape...>::get(uint64_t internalIndex) const
	{
		assert(internalIndex < _length);
		return _values[internalIndex];
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const TValue& StaticTensor<TValue, Shape...>::get(const uint64_t* indices) const
	{
		return get(getInternalIndex(indices));
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const TValue& StaticTensor<TValue, Shape...>::get(const std::initializer_list<uint64_t>& indices) const
	{
		assert(indices.size() == _order);
		return get(getInternalIndex(indices.begin()));
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::set(uint64_t internalIndex, const TValue& value)
	{
		assert(internalIndex < _length);
		_values[internalIndex] = value;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::set(const uint64_t* indices, const TValue& value)
	{
		set(getInternalIndex(indices), value);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::set(const std::initializer_list<uint64_t>& indices, const TValue& value)
	{
		assert(indices.size() == _order);
		set(getInternalIndex(indices.begin()), value);
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::getIndices(uint64_t internalIndex, uint64_t* indices) const
	{
		uint64_t factor = _length;

		for (uint64_t i = 0; i < _order; ++i)
		{
			factor /= _shape[i];
			const uint64_t index = internalIndex / factor;
			indices[i] = index;
			internalIndex -= index * factor;
		}
	}

	template<typename TValue, uint64_t... Shape>
	constexpr uint64_t StaticTensor<TValue, Shape...>::getInternalIndex(const uint64_t* indices) const
	{
		uint64_t index = 0;
		for (uint64_t i = 0; i < _order; ++i)
		{
			assert(indices[i] < _shape[i]);
			index = index * _shape[i] + indices[i];
		}

		return index;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr uint64_t StaticTensor<TValue, Shape...>::getInternalLength() const
	{
		return _length;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr uint64_t StaticTensor<TValue, Shape...>::getOrder() const
	{
		return _order;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const uint64_t* StaticTensor<TValue, Shape...>::getShape() const
	{
		return _shape;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr uint64_t StaticTensor<TValue, Shape...>::getSize(uint64_t i) const
	{
		assert(i < _order);
		return _shape[i];
	}

	template<typename TValue, uint64_t... Shape>
	constexpr TValue* StaticTensor<TValue, Shape...>::getData()
	{
		return _values;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr const TValue* StaticTensor<TValue, Shape...>::getData() const
	{
		return _values;
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::create(uint64_t order, const uint64_t* shape)
	{
		assert(order == _order);
		assert(std::equal(_shape, _shape + _order, shape));
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::copyFrom(const TensorBase<TValue>& tensor)
	{
		const StaticTensor<TValue, Shape...>* staticTensor = dynamic_cast<const StaticTensor<TValue, Shape...>*>(&tensor);
		if (staticTensor)
		{
			std::copy(staticTensor->_values, staticTensor->_values + _length, _values);
		}
		else
		{
			DenseTensor<TValue>::copyFrom(tensor);
		}
	}

	template<typename TValue, uint64_t... Shape>
	constexpr void StaticTensor<TValue, Shape...>::destroy()
	{
	}


	template<typename TValue, uint64_t... Shape>
	TensorIterator<sizeof...(Shape)> begin(const StaticTensor<TValue, Shape...>& tensor)
	{
		return TensorIterator<sizeof...(Shape)>(&tensor, false);
	}

	template<typename TValue, uint64_t... Shape>
	TensorIterator<sizeof...(Shape)> end(const StaticTensor<TValue, Shape...>& tensor)
	{
		return TensorIterator<sizeof...(Shape)>(&tensor, true);
	}


	template<typename TValue, uint64_t... ShapeA, uint64_t... ShapeB>
	StaticTensor<TValue, ShapeA..., ShapeB...> tensorProduct(const StaticTensor<TValue, ShapeA...>& a, const StaticTensor<TValue, ShapeB...>& b)
	{
		StaticTensor<TValue, ShapeA..., ShapeB...> result;
		a.tensorProduct(b, result);
		return result;
	}

	template<typename TValue, uint64_t... ShapeA, uint64_t N, uint64_t... ShapeB>
	StaticTensor<TValue, ShapeA..., ShapeB...> contractedTensorProduct(const StaticTensor<TValue, ShapeA..., N>& a, const StaticTensor<TValue, N, ShapeB...>& b)
	{
		StaticTensor<TValue, ShapeA..., ShapeB...> result;
		a.contractedTensorProduct(b, result);
		return result;
	}


	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix() : DenseMatrix<StaticTensor<TValue, NRow, NCol>>()
	{
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix(const TValue& value) : DenseMatrix<StaticTensor<TValue, NRow, NCol>>(value)
	{
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix(const TValue* values) : DenseMatrix<StaticTensor<TValue, NRow, NCol>>(values)
	{
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix(const std::vector<TValue>& values) : DenseMatrix<StaticTensor<TValue, NRow, NCol>>(values)
	{
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix(const std::initializer_list<TValue>& values) : DenseMatrix<StaticTensor<TValue, NRow, NCol>>(values)
	{
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	constexpr StaticMatrix<TValue, NRow, NCol>::StaticMatrix(const TensorBase<TValue>& tensor) : DenseMatrix<StaticTensor<TValue, NRow, NCol>>(tensor)
	{
	}


	template<typename TValue, uint64_t NRow, uint64_t NMiddle, uint64_t NCol>
	StaticMatrix<TValue, NRow, NCol> operator*(const StaticMatrix<TValue, NRow, NMiddle>& a, const StaticMatrix<TValue, NMiddle, NCol>& b)
	{
		StaticMatrix<TValue, NRow, NCol> result;
		a.matrixProduct(b, result);
		return result;
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticVector<TValue, NRow> operator*(const StaticMatrix<TValue, NRow, NCol>& matrix, const StaticVector<TValue, NCol>& vector)
	{
		StaticVector<TValue, NRow> result;
		matrix.vectorProduct(vector, result);
		return result;
	}

	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticMatrix<TValue, NCol, NRow> transpose(const StaticMatrix<TValue, NRow, NCol>& matrix)
	{
		StaticMatrix<TValue, NCol, NRow> result;
		matrix.transpose(result);
		return result;
	}


	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector() : DenseTensor<StaticTensor<TValue, Size>>()
	{
	}

	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector(const TValue& value) : DenseVector<StaticTensor<TValue, Size>>(value)
	{
	}

	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector(const TValue* values) : DenseVector<StaticTensor<TValue, Size>>(values)
	{
	}

	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector(const std::vector<TValue>& values) : DenseVector<StaticTensor<TValue, Size>>(values)
	{
	}

	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector(const std::initializer_list<TValue>& values) : DenseVector<StaticTensor<TValue, Size>>(values)
	{
	}

	template<typename TValue, uint64_t Size>
	constexpr StaticVector<TValue, Size>::StaticVector(const TensorBase<TValue>& tensor) : DenseVector<StaticTensor<TValue, Size>>(tensor)
	{
	}


	template<typename TValue, uint64_t NRow, uint64_t NCol>
	StaticVector<TValue, NCol> operator*(const StaticVector<TValue, NRow>& vector, const StaticMatrix<TValue, NRow, NCol>& matrix)
	{
		StaticVector<TValue, NCol> result;
		vector.matrixProduct(matrix, result);
		return result;
	}
}
