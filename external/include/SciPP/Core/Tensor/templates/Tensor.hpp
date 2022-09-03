#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/Tensor.hpp>

namespace scp
{
	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor() :
		_values(nullptr),
		_tree(nullptr),
		_shape(nullptr),
		_treeOffset(1),
		_length(0),
		_owner(true)
	{
		static_assert(Order > 0);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const uint64_t* shape) : Tensor<TValue, Order>()
	{
		create(Order, shape);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const uint64_t* shape, const TValue& value) : Tensor<TValue, Order>(shape)
	{
		std::fill(_values, _values + _length, value);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const uint64_t* shape, const TValue* values) : Tensor<TValue, Order>(shape)
	{
		std::copy(values, values + _length, _values);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const uint64_t* shape, const std::vector<TValue>& values) : Tensor<TValue, Order>(shape, values.data())
	{
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const uint64_t* shape, const std::initializer_list<TValue>& values) : Tensor<TValue, Order>(shape, values.begin())
	{
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::vector<uint64_t>& shape) : Tensor<TValue, Order>(shape.data())
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::vector<uint64_t>& shape, const TValue& value) : Tensor<TValue, Order>(shape.data(), value)
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::vector<uint64_t>& shape, const TValue* values) : Tensor<TValue, Order>(shape.data(), values)
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values) : Tensor<TValue, Order>(shape.data(), values.data())
	{
		assert(shape.size() == Order);
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values) : Tensor<TValue, Order>(shape.data(), values.begin())
	{
		assert(shape.size() == Order);
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::initializer_list<uint64_t>& shape) : Tensor<TValue, Order>(shape.begin())
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::initializer_list<uint64_t>& shape, const TValue& value) : Tensor<TValue, Order>(shape.begin(), value)
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::initializer_list<uint64_t>& shape, const TValue* values) : Tensor<TValue, Order>(shape.begin(), values)
	{
		assert(shape.size() == Order);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values) : Tensor<TValue, Order>(shape.begin(), values.data())
	{
		assert(shape.size() == Order);
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values) : Tensor<TValue, Order>(shape.begin(), values.begin())
	{
		assert(shape.size() == Order);
		assert(values.size() == _length);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const TensorBase<TValue>& tensor) : Tensor<TValue, Order>()
	{
		create(tensor.getOrder(), tensor.getShape());
		copyFrom(tensor);
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(const Tensor<TValue, Order>& tensor) : Tensor<TValue, Order>(dynamic_cast<const TensorBase<TValue>&>(tensor))
	{
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::Tensor(Tensor<TValue, Order>&& tensor) : Tensor<TValue, Order>()
	{
		moveFrom(std::move(tensor));
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>& Tensor<TValue, Order>::operator=(const Tensor<TValue, Order>& tensor)
	{
		copyFrom(tensor);
		return *this;
	}
	
	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>& Tensor<TValue, Order>::operator=(Tensor<TValue, Order>&& tensor)
	{
		moveFrom(std::move(tensor));
		return *this;
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>* Tensor<TValue, Order>::clone() const
	{
		return new Tensor<TValue, Order>(*this);
	}

	template<typename TValue, uint64_t Order>
	constexpr typename Tensor<TValue, Order>::SubTensor& Tensor<TValue, Order>::operator[](uint64_t i)
	{
		assert(i < _shape[0]);
		return _tree[i * _treeOffset];
	}

	template<typename TValue, uint64_t Order>
	constexpr const typename Tensor<TValue, Order>::SubTensor& Tensor<TValue, Order>::operator[](uint64_t i) const
	{
		assert(i < _shape[0]);
		return _tree[i * _treeOffset];
	}

	namespace _scp
	{
		template<typename TValue, uint64_t Order>
		void cooleyTukey(Tensor<TValue, Order>& a, const std::array<std::vector<TValue>, Order>& exponentials, const uint64_t* shape, const uint64_t* offset, uint64_t* stride)
		{
			// Directly exit if it is a single variable

			{
				uint64_t i = 0;
				for (; i < Order; ++i)
				{
					if (shape[i] != 1)
					{
						break;
					}
				}

				if (i == Order)
				{
					return;
				}
			}

			// Useful variables

			uint64_t cellShape[Order];
			for (uint64_t i = 0; i < Order; ++i)
			{
				cellShape[i] = shape[i];
				for (uint64_t j = 2; j < shape[i]; ++j)
				{
					if (shape[i] % j == 0)
					{
						cellShape[i] = j;
						break;
					}
				}
			}

			uint64_t subShape[Order];
			for (uint64_t i = 0; i < Order; ++i)
			{
				subShape[i] = shape[i] / cellShape[i];
			}

			uint64_t subOffset[Order];
			for (uint64_t i = 0; i < Order; ++i)
			{
				subOffset[i] = offset[i] * cellShape[i];
			}

			// Compute fft of each sub tensor

			TensorSweep<Order> cellSweep(cellShape);
			for (const TensorPosition& aCellPos : cellSweep)
			{
				for (uint64_t i = 0; i < Order; ++i)
				{
					stride[i] += aCellPos.indices[i] * offset[i];
				}

				cooleyTukey(a, exponentials, subShape, subOffset, stride);

				for (uint64_t i = 0; i < Order; ++i)
				{
					stride[i] -= aCellPos.indices[i] * offset[i];
				}
			}

			// Store the coefficients calculated before merging them

			Tensor<TValue, Order> tmp(shape);
			uint64_t indices[Order];
			for (const TensorPosition& pos : tmp)
			{
				for (uint64_t i = 0; i < Order; ++i)
				{
					indices[i] = stride[i] + pos.indices[i] * offset[i];
				}

				tmp.set(pos.indices, a.get(indices));
			}

			// Iterate over a sub tensor and merge each cell asociated

			uint64_t aCellIndices[Order];
			uint64_t tmpCellIndices[Order];
			uint64_t aIndices[Order];
			uint64_t tmpIndices[Order];
			TensorSweep<Order> subSweep(subShape);
			for (const TensorPosition& subPos : subSweep)
			{
				// Compute the position of the corner of the cell on 'a' and on 'tmp'

				for (uint64_t i = 0; i < Order; ++i)
				{
					aCellIndices[i] = stride[i] + subPos.indices[i] * offset[i];
					tmpCellIndices[i] = subPos.indices[i] * cellShape[i];
				}

				// Merge the elements of the cell of 'tmp' into the elements of the cell of 'a'

				for (const TensorPosition& aCellPos : cellSweep)
				{
					for (uint64_t i = 0; i < Order; ++i)
					{
						aIndices[i] = aCellIndices[i] + aCellPos.indices[i] * subShape[i] * offset[i];
					}

					TValue value = 0;

					for (const TensorPosition& tmpCellPos : cellSweep)
					{
						for (uint64_t i = 0; i < Order; ++i)
						{
							tmpIndices[i] = tmpCellIndices[i] + tmpCellPos.indices[i];
						}

						TValue factor = 1;
						for (uint64_t i = 0; i < Order; ++i)
						{
							factor *= exponentials[i][((subPos.indices[i] * tmpCellPos.indices[i]) % shape[i]) * offset[i]];
							factor *= exponentials[i][((tmpCellPos.indices[i] * aCellPos.indices[i]) % cellShape[i]) * subShape[i] * offset[i]];
						}

						value += factor * tmp.get(tmpIndices);
					}

					a.set(aIndices, value);
				}
			}
		}
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>& Tensor<TValue, Order>::fft()
	{
		if constexpr (_scp::IsComplex<TValue>::value)
		{
			uint64_t stride[Order];
			std::fill(stride, stride + Order, 0);

			uint64_t offset[Order];
			std::fill<uint64_t*, uint64_t>(offset, offset + Order, 1);

			std::array<std::vector<TValue>, Order> exponentials;
			for (uint64_t i = 0; i < Order; ++i)
			{
				exponentials[i].resize(_shape[i]);
				for (uint64_t j = 0; j < _shape[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, -2 * std::numbers::pi * j / _shape[i]));
				}
			}

			_scp::cooleyTukey(*this, exponentials, _shape, offset, stride);
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>& Tensor<TValue, Order>::ifft()
	{
		if constexpr (_scp::IsComplex<TValue>::value)
		{
			uint64_t stride[Order];
			std::fill(stride, stride + Order, 0);

			uint64_t offset[Order];
			std::fill<uint64_t*, uint64_t>(offset, offset + Order, 1);

			std::array<std::vector<TValue>, Order> exponentials;
			for (uint64_t i = 0; i < Order; ++i)
			{
				exponentials[i].resize(_shape[i]);
				for (uint64_t j = 0; j < _shape[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, 2 * std::numbers::pi * j / _shape[i]));
				}
			}

			_scp::cooleyTukey(*this, exponentials, _shape, offset, stride);

			*this /= _length;
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue, uint64_t Order>
	constexpr const TValue& Tensor<TValue, Order>::get(uint64_t internalIndex) const
	{
		assert(internalIndex < _length);
		return _values[internalIndex];
	}

	template<typename TValue, uint64_t Order>
	constexpr const TValue& Tensor<TValue, Order>::get(const uint64_t* indices) const
	{
		return get(getInternalIndex(indices));
	}

	template<typename TValue, uint64_t Order>
	constexpr const TValue& Tensor<TValue, Order>::get(const std::initializer_list<uint64_t>& indices) const
	{
		assert(indices.size() == Order);
		return get(getInternalIndex(indices.begin()));
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::set(uint64_t internalIndex, const TValue& value)
	{
		assert(internalIndex < _length);
		_values[internalIndex] = value;
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::set(const uint64_t* indices, const TValue& value)
	{
		set(getInternalIndex(indices), value);
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::set(const std::initializer_list<uint64_t>& indices, const TValue& value)
	{
		assert(indices.size() == Order);
		set(getInternalIndex(indices.begin()), value);
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::getIndices(uint64_t internalIndex, uint64_t* indices) const
	{
		uint64_t factor = _length;
		for (uint64_t i = 0; i < Order; ++i)
		{
			factor /= _shape[i];
			const uint64_t index = internalIndex / factor;
			indices[i] = index;
			internalIndex -= index * factor;
		}
	}

	template<typename TValue, uint64_t Order>
	constexpr uint64_t Tensor<TValue, Order>::getInternalIndex(const uint64_t* indices) const
	{
		uint64_t index = 0;
		for (uint64_t i = 0; i < Order; ++i)
		{
			assert(indices[i] < _shape[i]);
			index = index * _shape[i] + indices[i];
		}

		return index;
	}

	template<typename TValue, uint64_t Order>
	constexpr uint64_t Tensor<TValue, Order>::getInternalLength() const
	{
		return _length;
	}

	template<typename TValue, uint64_t Order>
	constexpr uint64_t Tensor<TValue, Order>::getOrder() const
	{
		return Order;
	}

	template<typename TValue, uint64_t Order>
	constexpr const uint64_t* Tensor<TValue, Order>::getShape() const
	{
		return _shape;
	}

	template<typename TValue, uint64_t Order>
	constexpr uint64_t Tensor<TValue, Order>::getSize(uint64_t i) const
	{
		assert(i < Order);
		return _shape[i];
	}

	template<typename TValue, uint64_t Order>
	constexpr uint64_t Tensor<TValue, Order>::getTotalLength() const
	{
		return _length;
	}

	template<typename TValue, uint64_t Order>
	constexpr TValue* Tensor<TValue, Order>::getData()
	{
		return _values;
	}

	template<typename TValue, uint64_t Order>
	constexpr const TValue* Tensor<TValue, Order>::getData() const
	{
		return _values;
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::initSubTensor(TValue* values, uint64_t* shape, uint64_t length, uint64_t treeLength)
	{
		_values = values;
		if constexpr (Order == 1)
		{
			_tree = values;
			_treeOffset = 1;
		}
		else
		{
			_tree = reinterpret_cast<SubTensor*>(this + 1);
			_treeOffset = treeLength / shape[0];

			uint64_t subLength = length / shape[0];
			for (uint64_t i = 0; i < shape[0]; ++i)
			{
				_tree[i * _treeOffset].initSubTensor(_values + i * subLength, shape + 1, subLength, _treeOffset - 1);
			}
		}
		_shape = shape;
		_length = length;
		_owner = false;
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::create(uint64_t order, const uint64_t* shape)
	{
		assert(order == Order);
		assert(_values == nullptr);
		assert(_owner);

		_shape = new uint64_t[Order];
		std::copy(shape, shape + Order, _shape);

		_length = std::accumulate(_shape, _shape + Order, 1, std::multiplies<uint64_t>());
		assert(_length != 0);

		_values = new TValue[_length];

		if constexpr (Order == 1)
		{
			_tree = _values;
			_treeOffset = 1;
		}
		else
		{
			uint64_t treeLength = 0;
			const uint64_t* it = _shape + Order - 2;
			const uint64_t* const itEnd = _shape - 1;
			while (it != itEnd)
			{
				treeLength = (treeLength + 1) * *it;
				--it;
			}

			_tree = new SubTensor[treeLength];
			_treeOffset = treeLength / _shape[0];

			uint64_t subLength = _length / _shape[0];
			for (uint64_t i = 0; i < _shape[0]; ++i)
			{
				_tree[i * _treeOffset].initSubTensor(_values + i * subLength, _shape + 1, subLength, _treeOffset - 1);
			}
		}
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::copyFrom(const TensorBase<TValue>& tensor)
	{
		assert(_owner || (tensor.getOrder() == Order && std::equal(_shape, _shape + Order, tensor.getShape())));
		DenseTensor<TValue>::copyFrom(tensor);
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::moveFrom(TensorBase<TValue>&& tensor)
	{
		Tensor<TValue, Order>* spTensor = dynamic_cast<Tensor<TValue, Order>*>(&tensor);
		if (spTensor)
		{
			if (_owner && spTensor->_owner)
			{
				destroy();

				_values = spTensor->_values;
				_tree = spTensor->_tree;
				_shape = spTensor->_shape;
				_treeOffset = spTensor->_treeOffset;
				_length = spTensor->_length;

				spTensor->_values = nullptr;
				spTensor->_tree = nullptr;
				spTensor->_shape = nullptr;
				spTensor->_treeOffset = 1;
				spTensor->_length = 0;
			}
			else
			{
				copyFrom(tensor);
			}
		}
		else
		{
			DenseTensor<TValue>::moveFrom(std::move(tensor));
		}
	}

	template<typename TValue, uint64_t Order>
	constexpr void Tensor<TValue, Order>::destroy()
	{
		if (_owner && _values)
		{
			delete[] _values;
			delete[] _shape;
			if constexpr (Order != 1)
			{
				delete[] _tree;
			}

			_values = nullptr;
			_tree = nullptr;
			_shape = nullptr;
			_treeOffset = 1;
			_length = 1;
		}
	}

	template<typename TValue, uint64_t Order>
	constexpr Tensor<TValue, Order>::~Tensor()
	{
		destroy();
	}


	template<typename TValue, uint64_t Order>
	constexpr TensorIterator<Order> begin(const Tensor<TValue, Order>& tensor)
	{
		return TensorIterator<Order>(&tensor, false);
	}

	template<typename TValue, uint64_t Order>
	constexpr TensorIterator<Order> end(const Tensor<TValue, Order>& tensor)
	{
		return TensorIterator<Order>(&tensor, true);
	}


	template<typename TValue, uint64_t Order>
	Tensor<TValue, Order - 2> tensorContraction(const Tensor<TValue, Order>& tensor, uint64_t i, uint64_t j)
	{
		static_assert(Order > 2);
		assert(i != j);
		assert(tensor.getSize(i) == tensor.getSize(j));

		if (i > j)
		{
			std::swap(i, j);
		}

		const uint64_t* tensorShape = tensor.getShape();
		uint64_t shape[Order - 2];
		if (i != 0)
		{
			std::copy(tensorShape, tensorShape + i, shape);
		}
		if (i != j - 1)
		{
			std::copy(tensorShape + i + 1, tensorShape + j, shape + i);
		}
		if (j != Order - 1)
		{
			std::copy(tensorShape + j + 1, tensorShape + Order, shape + j - 1);
		}

		Tensor<TValue, Order - 2> contraction(shape);
		tensor.tensorContraction(i, j, contraction);

		return contraction;
	}

	template<typename TValue, uint64_t OrderA, uint64_t OrderB>
	Tensor<TValue, OrderA + OrderB> tensorProduct(const Tensor<TValue, OrderA>& a, const Tensor<TValue, OrderB>& b)
	{
		const uint64_t* shapeA = a.getShape();
		const uint64_t* shapeB = b.getShape();

		uint64_t shape[OrderA + OrderB];
		std::copy(shapeA, shapeA + OrderA, shape);
		std::copy(shapeB, shapeB + OrderB, shape + OrderA);

		Tensor<TValue, OrderA + OrderB> c(shape);
		a.tensorProduct(b, c);

		return c;
	}

	template<typename TValue, uint64_t OrderA, uint64_t OrderB>
	Tensor<TValue, OrderA + OrderB - 2> contractedTensorProduct(const Tensor<TValue, OrderA>& a, const Tensor<TValue, OrderB>& b)
	{
		static_assert(OrderA + OrderB > 2);
		assert(a.getSize(OrderA - 1) == b.getSize(0));

		const uint64_t* shapeA = a.getShape();
		const uint64_t* shapeB = b.getShape();

		uint64_t shape[OrderA + OrderB - 2];
		std::copy(shapeA, shapeA + OrderA - 1, shape);
		std::copy(shapeB, shapeB + OrderB - 1, shape + OrderA - 1);

		Tensor<TValue, OrderA + OrderB - 2> c(shape);
		a.contractedTensorProduct(b, c);

		return c;
	}


	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(uint64_t row, uint64_t col) : DenseMatrix<Tensor<TValue, 2>>(std::vector<uint64_t>{ { row, col } })
	{
	}

	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(uint64_t row, uint64_t col, const TValue& value) : DenseMatrix<Tensor<TValue, 2>>(std::vector<uint64_t>{ { row, col } }, value)
	{
	}

	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(uint64_t row, uint64_t col, const TValue* values) : DenseMatrix<Tensor<TValue, 2>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(uint64_t row, uint64_t col, const std::vector<TValue>& values) : DenseMatrix<Tensor<TValue, 2>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values) : DenseMatrix<Tensor<TValue, 2>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr Matrix<TValue>::Matrix(const TensorBase<TValue>& tensor) : DenseMatrix<Tensor<TValue, 2>>(tensor)
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>& Matrix<TValue>::operator[](uint64_t i)
	{
		return static_cast<Vector<TValue>&>(Tensor<TValue, 2>::operator[](i));
	}

	template<typename TValue>
	constexpr const Vector<TValue>& Matrix<TValue>::operator[](uint64_t i) const
	{
		return static_cast<const Vector<TValue>&>(Tensor<TValue, 2>::operator[](i));
	}


	template<typename TValue>
	Matrix<TValue> operator*(const Matrix<TValue>& a, const Matrix<TValue>& b)
	{
		Matrix<TValue> result(a.getSize(0), b.getSize(1));
		a.matrixProduct(b, result);
		return result;
	}

	template<typename TValue>
	Vector<TValue> operator*(const Matrix<TValue>& matrix, const Vector<TValue>& vector)
	{
		Vector<TValue> result(matrix.getSize(0));
		matrix.vectorProduct(vector, result);
		return result;
	}

	template<typename TValue>
	Matrix<TValue> transpose(const Matrix<TValue>& matrix)
	{
		Matrix<TValue> result(matrix.getSize(1), matrix.getSize(0));
		matrix.transpose(result);
		return result;
	}


	template<typename TValue>
	constexpr Vector<TValue>::Vector(uint64_t size) : DenseVector<Tensor<TValue, 1>>(&size)
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>::Vector(uint64_t size, const TValue& value) : DenseVector<Tensor<TValue, 1>>(&size, value)
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>::Vector(uint64_t size, const TValue* values) : DenseVector<Tensor<TValue, 1>>(&size, values)
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>::Vector(const std::vector<TValue>& values) : DenseVector<Tensor<TValue, 1>>(std::vector<uint64_t>{ { values.size() } }, values.data())
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>::Vector(const std::initializer_list<TValue>& values) : DenseVector<Tensor<TValue, 1>>(std::vector<uint64_t>{ { values.size() } }, values.begin())
	{
	}

	template<typename TValue>
	constexpr Vector<TValue>::Vector(const TensorBase<TValue>& tensor) : DenseVector<Tensor<TValue, 1>>(tensor)
	{
	}

	
	template<typename TValue>
	Vector<TValue> operator*(const Vector<TValue>& vector, const Matrix<TValue>& matrix)
	{
		Vector<TValue> result(matrix.getSize(1));
		vector.matrixProduct(matrix, result);
		return result;
	}
}
