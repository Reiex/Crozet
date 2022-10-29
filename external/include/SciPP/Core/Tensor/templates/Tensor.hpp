///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/Core/Tensor/Tensor.hpp>

namespace scp
{
	SCP_TENSOR_DEF(template<typename TValue>, Tensor, Tensor<TValue>)

	template<typename TValue>
	constexpr Tensor<TValue>::Tensor() :
		_order(0),
		_length(0),
		_sizes(nullptr),
		_values(nullptr),
		_owner(true)
	{
	}

	template<typename TValue>
	constexpr Tensor<TValue>* Tensor<TValue>::createAroundMemory(uint64_t order, const uint64_t* sizes, TValue* memory)
	{
		assert(order != 0);
		assert(std::find(sizes, sizes + order, 0) == sizes + order);
		assert(memory);

		Tensor<TValue>* tensor = new Tensor<TValue>();

		tensor->_order = order;

		tensor->_length = std::accumulate(sizes, sizes + order, 1, std::multiplies<uint64_t>());

		tensor->_sizes = new uint64_t[order];
		std::copy_n(sizes, order, tensor->_sizes);

		tensor->_values = memory;

		tensor->_owner = false;

		return tensor;
	}

	template<typename TValue>
	constexpr Tensor<TValue>* Tensor<TValue>::createAroundMemory(const std::initializer_list<uint64_t>& sizes, TValue* memory)
	{
		return createAroundMemory(sizes.size(), sizes.begin(), memory);
	}

	template<typename TValue>
	constexpr Tensor<TValue>& Tensor<TValue>::fill(const TValue& value)
	{
		std::fill_n(_values, _length, value);
		return *this;
	}

	template<typename TValue>
	template<std::input_iterator TInput>
	constexpr Tensor<TValue>& Tensor<TValue>::copy(TInput it)
	{
		std::copy_n(it, _length, _values);
		return *this;
	}

	template<typename TValue>
	constexpr Tensor<TValue>& Tensor<TValue>::transform(const std::function<TValue(const TValue&)>& unaryOp)
	{
		std::transform(_values, _values + _length, _values, unaryOp);
		return *this;
	}

	template<typename TValue>
	template<std::input_iterator TInput>
	constexpr Tensor<TValue>& Tensor<TValue>::transform(TInput it, const std::function<TValue(const TValue&, const typename std::iterator_traits<TInput>::value_type&)>& binaryOp)
	{
		std::transform(_values, _values + _length, it, _values, binaryOp);
		return *this;
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr Tensor<TValue>& Tensor<TValue>::operator+=(const TTensor& tensor)
	{
		return transform(tensor.begin(), std::plus<TValue>());
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr Tensor<TValue>& Tensor<TValue>::operator-=(const TTensor& tensor)
	{
		return transform(tensor.begin(), std::minus<TValue>());
	}

	template<typename TValue>
	template<typename TScalar>
	constexpr Tensor<TValue>& Tensor<TValue>::operator*=(const TScalar& scalar)
	{
		return transform([&](const TValue& x) { return x * scalar; });
	}

	template<typename TValue>
	template<typename TScalar>
	constexpr Tensor<TValue>& Tensor<TValue>::operator/=(const TScalar& scalar)
	{
		return transform([&](const TValue& x) { return x / scalar; });
	}

	template<typename TValue>
	constexpr Tensor<TValue>& Tensor<TValue>::negate()
	{
		return transform(std::negate<TValue>());
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB>
	constexpr void Tensor<TValue>::tensorProduct(const TTensorA& tensorA, const TTensorB& tensorB)
	{
		const uint64_t orderA = tensorA.getOrder();
		const uint64_t orderB = tensorB.getOrder();
		const uint64_t* sizesA = tensorA.getSizes();
		const uint64_t* sizesB = tensorB.getSizes();

		assert(orderA + orderB == _order);
		assert(std::equal(sizesA, sizesA + orderA, _sizes));
		assert(std::equal(sizesB, sizesB + orderB, _sizes + orderA));

		const uint64_t length = tensorA.getElementCount();

		TValue* it = _values;
		for (uint64_t i = 0, j = 0, k = 0; i < _length; ++i, ++k, ++it)
		{
			if (k == length)
			{
				k = 0;
				++j;
			}

			*it = tensorA.get(j) * tensorB.get(k);
		}
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr Tensor<TValue>& Tensor<TValue>::hadamardProduct(const TTensor& tensor)
	{
		return transform(tensor.begin(), std::multiplies<TValue>());
	}

	template<typename TValue>
	constexpr Tensor<TValue>& Tensor<TValue>::fft()
	{
		if constexpr (IsComplex<TValue>)
		{
			uint64_t* stride = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
			std::fill(stride, stride + _order, 0);

			uint64_t* offset = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
			std::fill<uint64_t*, uint64_t>(offset, offset + _order, 1);

			TValue** exponentials = reinterpret_cast<TValue**>(alloca(_order * sizeof(TValue*)));
			for (uint64_t i = 0; i < _order; ++i)
			{
				exponentials[i] = reinterpret_cast<TValue*>(alloca(_sizes[i] * sizeof(TValue)));
				for (uint64_t j = 0; j < _sizes[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, -2 * std::numbers::pi * j / _sizes[i]));
				}
			}

			_cooleyTukey(exponentials, _sizes, offset, stride);
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue>
	constexpr Tensor<TValue>& Tensor<TValue>::ifft()
	{
		if constexpr (IsComplex<TValue>)
		{
			uint64_t* stride = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
			std::fill(stride, stride + _order, 0);

			uint64_t* offset = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
			std::fill<uint64_t*, uint64_t>(offset, offset + _order, 1);

			TValue** exponentials = reinterpret_cast<TValue**>(alloca(_order * sizeof(TValue*)));
			for (uint64_t i = 0; i < _order; ++i)
			{
				exponentials[i] = reinterpret_cast<TValue*>(alloca(_sizes[i] * sizeof(TValue)));
				for (uint64_t j = 0; j < _sizes[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, 2 * std::numbers::pi * j / _sizes[i]));
				}
			}

			_cooleyTukey(exponentials, _sizes, offset, stride);

			*this /= TValue(_length);
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue>
	template<BorderBehaviour BBehaviour, TensorConcept<TValue> TTensor>
	constexpr Tensor<TValue>& Tensor<TValue>::convolution(const TTensor& kernel)
	{
		assert(_order == kernel.getOrder());
		
		const TensorShape shape{ _order, _sizes };
		const TensorShape kernelShape{ kernel.getOrder(), kernel.getSizes() };
		
		// Do a copy of *this
		Tensor<TValue> tensor(*this);
		
		// Check that the kernel's sizes are odd
		for (uint64_t i = 0; i < _order; i++)
		{
			assert(kernelShape.sizes[i] % 2 == 1);
			assert(kernelShape.sizes[i] <= _sizes[i]);
		}
		
		// Compute offset (to center the kernel)
		int64_t* offset = reinterpret_cast<int64_t*>(alloca(_order * sizeof(int64_t)));
		for (uint64_t i = 0; i < _order; i++)
		{
			offset[i] = static_cast<int64_t>(kernelShape.sizes[i] / 2);
		}
		
		// For each element of the original tensor
		int64_t* offsetedIndices = reinterpret_cast<int64_t*>(alloca(_order * sizeof(int64_t)));
		for (const TensorPosition& pos : shape)
		{
			TValue value = 0;
		
			// For each element of the kernel
			for (const TensorPosition& kernelPos : kernelShape)
			{
				bool setToZero = false;
		
				int64_t* itOffsetedIndices = offsetedIndices;
				const uint64_t* itSizes = _sizes;
				const int64_t* itOffset = offset;
				const uint64_t* itIndices = pos.indices;
				const uint64_t* itKernelIndices = kernelPos.indices;
		
				// Compute the corresponding indices to poll
				for (uint64_t k = 0; k < _order; ++k, ++itKernelIndices, ++itOffsetedIndices, ++itSizes, ++itOffset, ++itIndices)
				{
					*itOffsetedIndices = static_cast<int64_t>(*itIndices) + *itOffset - static_cast<int64_t>(*itKernelIndices);
		
					if constexpr (BBehaviour == BorderBehaviour::Zero)
					{
						if (*itOffsetedIndices < 0 || *itOffsetedIndices >= *itSizes)
						{
							setToZero = true;
							break;
						}
					}
					else if constexpr (BBehaviour == BorderBehaviour::Continuous)
					{
						*itOffsetedIndices = std::clamp<int64_t>(*itOffsetedIndices, 0, *itSizes - 1);
					}
					else if constexpr (BBehaviour == BorderBehaviour::Periodic)
					{
						*itOffsetedIndices = (*itOffsetedIndices + *itSizes) % *itSizes;
					}
				}
		
				// Add the product to the result
				if (!setToZero)
				{
					value += tensor.get(reinterpret_cast<uint64_t*>(offsetedIndices)) * kernel.get(kernelPos.index);
				}
			}
		
			_values[pos.index] = value;
		}
		
		return *this;
	}

	namespace _scp
	{
		template<typename TValue, TensorConcept<TValue> TTensor, typename TScalar>
		constexpr TValue lerp(const TTensor& tensor, const uint64_t* sizesRev, uint64_t* indicesRev, const TScalar* coeffsRev, uint64_t nCoeffs)
		{
			if (nCoeffs == 0)
			{
				return tensor.get(indicesRev + 1);
			}
		
			const TValue x = _scp::lerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			++(*indicesRev);
			const TValue y = *indicesRev == *sizesRev ? x : _scp::lerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			--(*indicesRev);
		
			return *coeffsRev * y + (1.0 - *coeffsRev) * x;
		}
		
		template<typename TValue, TensorConcept<TValue> TTensor, typename TScalar>
		constexpr TValue cerp(const TTensor& tensor, const uint64_t* sizesRev, uint64_t* indicesRev, const TScalar* coeffsRev, uint64_t nCoeffs)
		{
			if (nCoeffs == 0)
			{
				return tensor.get(indicesRev + 1);
			}
		
			const TValue x1 = _scp::cerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			--(*indicesRev);
			const TValue x0 = *indicesRev == UINT64_MAX ? x1 : _scp::cerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			*indicesRev += 2;
			const TValue x2 = *indicesRev == *sizesRev ? x1 : _scp::cerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			TValue x3 = x2;
			if (*indicesRev != *sizesRev)
			{
				++(*indicesRev);
				x3 = *indicesRev == *sizesRev ? x2 : _scp::cerp<TValue>(tensor, sizesRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
				--(*indicesRev);
			}
			--(*indicesRev);
		
			const TValue a = -0.5*x0 + 1.5*x1 - 1.5*x2 + 0.5*x3;
			const TValue b =      x0 - 2.5*x1 + 2.0*x2 - 0.5*x3;
			const TValue c = -0.5*x0          + 0.5*x2;
			const TValue d =               x1;
		
			const TScalar& t = *coeffsRev;
		
			return d + t*(c + t*(b + t*a));
		}
	}

	template<typename TValue>
	template<typename TScalar, InterpolationMethod IMethod, TensorConcept<TValue> TTensor>
	constexpr void Tensor<TValue>::interpolation(const TTensor& tensor)
	{
		const TensorShape shape{ _order, _sizes };
		const TensorShape tensorShape{ tensor.getOrder(), tensor.getSizes() };
		
		assert(shape.order == tensorShape.order);
		
		uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		TScalar* coeffs = reinterpret_cast<TScalar*>(alloca(_order * sizeof(TScalar)));

		TScalar* sizesRatio = reinterpret_cast<TScalar*>(alloca(_order * sizeof(TScalar)));
		for (uint64_t i = 0; i < _order; ++i)
		{
			sizesRatio[i] = static_cast<TScalar>(tensorShape.sizes[i] - 1) / (_sizes[i] - 1);
		}
		
		const uint64_t* sizesRev = tensorShape.sizes + _order - 1;
		uint64_t* indicesRev = indices + _order - 1;
		const TScalar* coeffsRev = coeffs + _order - 1;
		
		for (const TensorPosition& pos : shape)
		{
			for (uint64_t i = 0; i < _order; ++i)
			{
				coeffs[i] = pos.indices[i] * sizesRatio[i];
				indices[i] = static_cast<uint64_t>(coeffs[i]);
				coeffs[i] -= indices[i];
			}
		
			if constexpr (IMethod == InterpolationMethod::Nearest)
			{
				_values[pos.index] = tensor.get(indices);
			}
			else if constexpr (IMethod == InterpolationMethod::Linear)
			{
				_values[pos.index] = _scp::lerp<TValue>(tensor, sizesRev, indicesRev, coeffsRev, _order);
			}
			else if constexpr (IMethod == InterpolationMethod::Cubic)
			{
				_values[pos.index] = _scp::cerp<TValue>(tensor, sizesRev, indicesRev, coeffsRev, _order);
			}
		}
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr void Tensor<TValue>::tensorContraction(const TTensor& tensor, uint64_t i, uint64_t j)
	{
		// TODO

		// const TensorShape shape{ getOrder(), getSizes() };
		// const TensorShape resultShape{ result.getOrder(), result.getSizes() };
		// 
		// assert(shape.order > 2);
		// assert(resultShape.order == shape.order - 2);
		// assert(i != j);
		// assert(shape.sizes[i] == shape.sizes[j]);
		// 
		// if (i > j)
		// {
		// 	std::swap(i, j);
		// }
		// 
		// assert(i == 0 || std::equal(shape.sizes, shape.sizes + i, resultShape.sizes));
		// assert(j == i + 1 || std::equal(shape.sizes + i + 1, shape.sizes + j, resultShape.sizes + i));
		// assert(j == shape.order - 1 || std::equal(shape.sizes + j + 1, shape.sizes + shape.order, resultShape.sizes + j - 1));
		// 
		// uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(shape.order * sizeof(uint64_t)));
		// for (const TensorPosition& pos : resultShape)
		// {
		// 	if (i != 0)
		// 	{
		// 		std::copy(pos.indices, pos.indices + i, indices);
		// 	}
		// 	if (i != j - 1)
		// 	{
		// 		std::copy(pos.indices + i, pos.indices + j - 1, indices + i + 1);
		// 	}
		// 	if (j != shape.order - 1)
		// 	{
		// 		std::copy(pos.indices + j - 1, pos.indices + resultShape.order, indices + j + 1);
		// 	}
		// 
		// 	TValue value = 0;
		// 	for (uint64_t k = 0; k < shape.sizes[i]; ++k)
		// 	{
		// 		indices[i] = k;
		// 		indices[j] = k;
		// 		value += get(indices);
		// 	}
		// 
		// 	result.set(pos.index, value);
		// }
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr bool Tensor<TValue>::operator==(const TTensor& tensor) const
	{
		if (_order != tensor.getOrder())
		{
			return false;
		}

		if (!std::equal(_sizes, _sizes + _order, tensor.getSizes()))
		{
			return false;
		}

		return std::equal(_values, _values + _length, tensor.begin());
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr bool Tensor<TValue>::operator!=(const TTensor& tensor) const
	{
		return !operator==(tensor);
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr TValue Tensor<TValue>::innerProduct(const TTensor& tensor) const
	{
		return std::inner_product(_values, _values + _length, tensor.begin(), TValue(0));
	}

	template<typename TValue>
	constexpr TValue Tensor<TValue>::norm() const
	{
		return std::inner_product(_values, _values + _length, _values, TValue(0));
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::minElement(const std::function<bool(const TValue&, const TValue&)>& compare) const
	{
		return *std::min_element(_values, _values + _length, compare);
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::maxElement(const std::function<bool(const TValue&, const TValue&)>& compare) const
	{
		return *std::max_element(_values, _values + _length, compare);
	}

	template<typename TValue>
	template<typename TScalar, InterpolationMethod IMethod>
	constexpr TValue Tensor<TValue>::getInterpolated(const TScalar* scalarIndices) const
	{
		uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		TScalar* coeffs = reinterpret_cast<TScalar*>(alloca(_order * sizeof(TScalar)));

		for (uint64_t i = 0; i < _order; ++i)
		{
			assert(scalarIndices[i] >= 0 && scalarIndices[i] <= _sizes[i] - 1);
			indices[i] = static_cast<uint64_t>(scalarIndices[i]);
			coeffs[i] = scalarIndices[i] - indices[i];
		}

		if constexpr (IMethod == InterpolationMethod::Nearest)
		{
			return get(indices);
		}
		else if constexpr (IMethod == InterpolationMethod::Linear)
		{
			return _scp::lerp<TValue>(*this, _sizes + _order - 1, indices + _order - 1, coeffs + _order - 1, _order);
		}
		else if constexpr (IMethod == InterpolationMethod::Cubic)
		{
			return _scp::cerp<TValue>(*this, _sizes + _order - 1, indices + _order - 1, coeffs + _order - 1, _order);
		}
	}

	template<typename TValue>
	template<typename TScalar, InterpolationMethod IMethod>
	constexpr TValue Tensor<TValue>::getInterpolated(const std::initializer_list<TScalar>& scalarIndices) const
	{
		assert(scalarIndices.size() == _order);
		return getInterpolated(scalarIndices.begin());
	}

	template<typename TValue>
	template<BorderBehaviour BBehaviour>
	constexpr const TValue& Tensor<TValue>::getOutOfBound(const int64_t* indices) const
	{
		uint64_t* realIndices = reinterpret_cast<uint64_t*>(alloca(sizeof(uint64_t) * _order));

		if constexpr (BBehaviour == BorderBehaviour::Zero)
		{
			static constexpr TValue zero = 0;

			bool isOut = false;
			for (uint64_t i = 0; !isOut && i < _order; ++i)
			{
				isOut = indices[i] < 0 || indices[i] >= _sizes[i];
			}

			if (isOut)
			{
				return zero;
			}
			else
			{
				std::copy_n(indices, _order, realIndices);
				return get(realIndices);
			}
		}
		else if constexpr (BBehaviour == BorderBehaviour::Continuous)
		{
			for (uint64_t i = 0; i < _order; ++i)
			{
				if (indices[i] < 0)
				{
					realIndices[i] = 0;
				}
				else if (indices[i] < _sizes[i])
				{
					realIndices[i] = indices[i];
				}
				else
				{
					realIndices[i] = _sizes[i] - 1;
				}
			}

			return get(realIndices);
		}
		else if constexpr (BBehaviour == BorderBehaviour::Periodic)
		{
			for (uint64_t i = 0; i < _order; ++i)
			{
				if (indices[i] < 0)
				{
					realIndices[i] = (indices[i] % _sizes[i]) + _sizes[i];
				}
				else if (indices[i] >= _sizes[i])
				{
					realIndices[i] = indices[i] % _sizes[i];
				}
				else
				{
					realIndices[i] = indices[i];
				}
			}

			return get(realIndices);
		}
	}

	template<typename TValue>
	template<BorderBehaviour BBehaviour>
	constexpr const TValue& Tensor<TValue>::getOutOfBound(const std::initializer_list<int64_t>& indices) const
	{
		assert(indices.size() == _order);
		return getOutOfBound<BBehaviour>(indices.begin());
	}

	template<typename TValue>
	constexpr TValue* Tensor<TValue>::begin()
	{
		return _values;
	}

	template<typename TValue>
	constexpr TValue* Tensor<TValue>::end()
	{
		return _values + _length;
	}

	template<typename TValue>
	constexpr const TValue* Tensor<TValue>::begin() const
	{
		return _values;
	}

	template<typename TValue>
	constexpr const TValue* Tensor<TValue>::end() const
	{
		return _values + _length;
	}

	template<typename TValue>
	constexpr const TValue* Tensor<TValue>::cbegin() const
	{
		return _values;
	}

	template<typename TValue>
	constexpr const TValue* Tensor<TValue>::cend() const
	{
		return _values + _length;
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::get(uint64_t index) const
	{
		return (*this)[index];
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::get(const uint64_t* indices) const
	{
		const TensorShape shape{ _order, _sizes };
		assert(shape.getIndex(indices) < _length);
		return _values[shape.getIndex(indices)];
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::get(const std::initializer_list<uint64_t>& indices) const
	{
		return (*this)[indices];
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::set(uint64_t index, const TValue& value)
	{
		(*this)[index] = value;
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::set(const uint64_t* indices, const TValue& value)
	{
		const TensorShape shape{ _order, _sizes };
		assert(shape.getIndex(indices) < _length);
		_values[shape.getIndex(indices)] = value;
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::set(const std::initializer_list<uint64_t>& indices, const TValue& value)
	{
		(*this)[indices] = value;
	}

	template<typename TValue>
	constexpr uint64_t Tensor<TValue>::getOrder() const
	{
		return _order;
	}

	template<typename TValue>
	constexpr const uint64_t* Tensor<TValue>::getSizes() const
	{
		return _sizes;
	}

	template<typename TValue>
	constexpr uint64_t Tensor<TValue>::getSize(uint64_t i) const
	{
		assert(i < _order);
		return _sizes[i];
	}

	template<typename TValue>
	constexpr uint64_t Tensor<TValue>::getElementCount() const
	{
		return _length;
	}

	template<typename TValue>
	constexpr TValue& Tensor<TValue>::operator[](uint64_t index)
	{
		assert(index < _length);
		return _values[index];
	}

	template<typename TValue>
	constexpr TValue& Tensor<TValue>::operator[](const std::initializer_list<uint64_t>& indices)
	{
		const TensorShape shape{ _order, _sizes };
		assert(shape.getIndex(indices.begin()) < _length);
		return _values[shape.getIndex(indices.begin())];
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::operator[](uint64_t index) const
	{
		assert(index < _length);
		return _values[index];
	}

	template<typename TValue>
	constexpr const TValue& Tensor<TValue>::operator[](const std::initializer_list<uint64_t>& indices) const
	{
		const TensorShape shape{ _order, _sizes };
		assert(shape.getIndex(indices.begin()) < _length);
		return _values[shape.getIndex(indices.begin())];
	}

	template<typename TValue>
	constexpr TValue& Tensor<TValue>::get(uint64_t index)
	{
		return (*this)[index];
	}

	template<typename TValue>
	constexpr TValue& Tensor<TValue>::get(const uint64_t* indices)
	{
		const TensorShape shape{ _order, _sizes };
		assert(shape.getIndex(indices) < _length);
		return _values[shape.getIndex(indices)];
	}

	template<typename TValue>
	constexpr TValue& Tensor<TValue>::get(const std::initializer_list<uint64_t>& indices)
	{
		return (*this)[indices];
	}

	template<typename TValue>
	constexpr TValue* Tensor<TValue>::getData()
	{
		return _values;
	}

	template<typename TValue>
	constexpr const TValue* Tensor<TValue>::getData() const
	{
		return _values;
	}

	template<typename TValue>
	constexpr Tensor<TValue>::~Tensor()
	{
		_destroy();
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::_create(uint64_t order, const uint64_t* sizes)
	{
		assert(order != 0);
		assert(std::find(sizes, sizes + order, 0) == sizes + order);

		_order = order;

		_length = std::accumulate(sizes, sizes + order, 1, std::multiplies<uint64_t>());

		_sizes = new uint64_t[_order];
		std::copy_n(sizes, _order, _sizes);

		_values = new TValue[_length];

		_owner = true;
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr void Tensor<TValue>::_copyFrom(const TTensor& tensor)
	{
		const uint64_t tensorOrder = tensor.getOrder();
		const uint64_t* tensorSizes = tensor.getSizes();

		if (_order != tensorOrder || !std::equal(_sizes, _sizes + _order, tensorSizes))
		{
			_destroy();
			_create(tensorOrder, tensorSizes);
		}

		std::copy_n(tensor.begin(), _length, _values);
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr void Tensor<TValue>::_moveFrom(TTensor&& tensor)
	{
		if constexpr (!std::derived_from<TTensor, Tensor<TValue>>)
		{
			_copyFrom(tensor);
			return;
		}

		if (!_owner || !tensor._owner)
		{
			_copyFrom(tensor);
			return;
		}

		_destroy();

		_order = tensor._order;
		_length = tensor._length;
		_sizes = tensor._sizes;
		_values = tensor._values;

		tensor._values = nullptr;
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::_destroy()
	{
		if (_values)
		{
			if (_owner)
			{
				delete[] _values;
			}

			delete[] _sizes;
		}

		_order = 0;
		_length = 0;
		_sizes = nullptr;
		_values = nullptr;
		_owner = true;
	}

	template<typename TValue>
	constexpr void Tensor<TValue>::_cooleyTukey(const TValue* const* exponentials, const uint64_t* sizes, uint64_t* offset, uint64_t* stride)
	{
		const TensorShape shape{ _order, sizes };
		
		// Directly exit if it is a single variable
		
		if (std::all_of(sizes, sizes + _order, [&](const uint64_t& x) { return x == 1; }))
		{
			return;
		}
		
		// Useful variables
		
		uint64_t* cellSizes = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		for (uint64_t i = 0; i < _order; ++i)
		{
			cellSizes[i] = sizes[i];
			for (uint64_t j = 2; j < sizes[i]; ++j)
			{
				if (sizes[i] % j == 0)
				{
					cellSizes[i] = j;
					break;
				}
			}
		}
		const TensorShape cellShape{ _order, cellSizes };
		const TensorShapeIterator cellShapeBegin = cellShape.begin();
		const TensorShapeIterator cellShapeEnd = cellShape.end();
		
		uint64_t* subSizes = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		for (uint64_t i = 0; i < _order; ++i)
		{
			subSizes[i] = sizes[i] / cellSizes[i];
		}
		const TensorShape subShape{ _order, subSizes };
		
		// Compute fft of each sub tensor

		for (uint64_t i = 0; i < _order; ++i)
		{
			offset[i] *= cellSizes[i];
		}
		
		for (const TensorPosition& pos : cellShape)
		{
			for (uint64_t i = 0; i < _order; ++i)
			{
				stride[i] += pos.indices[i] * offset[i] / cellSizes[i];
			}
		
			_cooleyTukey(exponentials, subSizes, offset, stride);
		
			for (uint64_t i = 0; i < _order; ++i)
			{
				stride[i] -= pos.indices[i] * offset[i] / cellSizes[i];
			}
		}

		for (uint64_t i = 0; i < _order; ++i)
		{
			offset[i] /= cellSizes[i];
		}
		
		// Store the coefficients calculated before merging them
		
		uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		uint64_t* cellIndices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		uint64_t* tmpCellIndices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));
		uint64_t* tmpIndices = reinterpret_cast<uint64_t*>(alloca(_order * sizeof(uint64_t)));

		Tensor<TValue> tmp(_order, sizes);
		for (const TensorPosition& pos : shape)
		{
			for (uint64_t i = 0; i < _order; ++i)
			{
				indices[i] = stride[i] + pos.indices[i] * offset[i];
			}

			tmp._values[pos.index] = get(indices);
		}
		
		// Iterate over a sub tensor and merge each cell asociated
		
		TensorShapeIterator cellShapeIterator = cellShapeBegin;
		TensorShapeIterator tmpCellShapeIterator = cellShapeBegin;
		const TensorPosition& cellPos = *cellShapeIterator;
		const TensorPosition& tmpCellPos = *tmpCellShapeIterator;
		for (const TensorPosition& subPos : subShape)
		{
			// Compute the position of the corner of the cell on '*this' and on 'tmp'
		
			for (uint64_t i = 0; i < _order; ++i)
			{
				cellIndices[i] = stride[i] + subPos.indices[i] * offset[i];
				tmpCellIndices[i] = subPos.indices[i] * cellSizes[i];
			}
		
			// Merge the elements of the cell of 'tmp' into the elements of the cell of '*this'
		
			for (cellShapeIterator = cellShapeBegin; cellShapeIterator != cellShapeEnd; ++cellShapeIterator)
			{
				for (uint64_t i = 0; i < _order; ++i)
				{
					indices[i] = cellIndices[i] + cellPos.indices[i] * subSizes[i] * offset[i];
				}
		
				TValue value = 0;

				for (tmpCellShapeIterator = cellShapeBegin; tmpCellShapeIterator != cellShapeEnd; ++tmpCellShapeIterator)
				{
					for (uint64_t i = 0; i < _order; ++i)
					{
						tmpIndices[i] = tmpCellIndices[i] + tmpCellPos.indices[i];
					}
		
					TValue factor = 1;
					for (uint64_t i = 0; i < _order; ++i)
					{
						factor *= exponentials[i][((subPos.indices[i] * tmpCellPos.indices[i]) % sizes[i]) * offset[i]];
						factor *= exponentials[i][((tmpCellPos.indices[i] * cellPos.indices[i]) % cellSizes[i]) * subSizes[i] * offset[i]];
					}
		
					value += factor * tmp.get(tmpIndices);
				}
		
				set(indices, value);
			}
		}
	}

	
	SCP_MATRIX_DEF(template<typename TValue>, Matrix, Matrix<TValue>)

	template<typename TValue>
	constexpr Matrix<TValue>* Matrix<TValue>::createAroundMemory(uint64_t row, uint64_t col, TValue* memory)
	{
		assert(row != 0 && col != 0);
		assert(memory);

		Matrix<TValue>* matrix = new Matrix<TValue>();

		matrix->_order = 2;

		matrix->_length = row * col;

		matrix->_sizes = new uint64_t[2];
		matrix->_sizes[0] = row;
		matrix->_sizes[1] = col;

		matrix->_values = memory;

		matrix->_owner = false;

		return matrix;
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB>
	constexpr void Matrix<TValue>::matrixProduct(const TTensorA& matrixA, const TTensorB& matrixB)
	{
		const uint64_t* sizesA = matrixA.getSizes();
		const uint64_t* sizesB = matrixB.getSizes();

		assert(matrixA.getOrder() == 2);
		assert(matrixB.getOrder() == 2);
		assert(sizesA[1] == sizesB[0]);
		assert(sizesA[0] == _sizes[0]);
		assert(sizesB[1] == _sizes[1]);

		TValue* it = _values;
		uint64_t a, b;
		for (uint64_t i = 0; i < _length; ++i, ++it)
		{
			*it = 0;

			b = i % sizesA[0];
			a = i - b;
			for (uint64_t k = 0; k < sizesA[1]; ++k, ++a, b += sizesB[1])
			{
				*it += matrixA.get(a) * matrixB.get(b);
			}
		}
	}

	template<typename TValue>
	constexpr Matrix<TValue>& Matrix<TValue>::transpose()
	{
		if (_sizes[0] == _sizes[1])
		{
			const uint64_t& size = _sizes[0];

			TValue* itA = _values;
			TValue* itB = _values;

			for (uint64_t j = 0; j < size; ++j)
			{
				itA += j + 1;
				itB += (j + 1) * size;

				for (uint64_t i = j + 1; i < size; ++i, ++itA, itB += size)
				{
					std::swap(*itA, *itB);
				}

				itB -= _length - 1;
			}
		}
		else
		{
			std::vector<bool> visited(_length, false);
			visited[0] = true;
			visited[_length - 1] = true;

			uint64_t n = 0;
			uint64_t i, j;
			TValue tmp;
			while (n != _length)
			{
				if (visited[n])
				{
					++n;
					continue;
				}

				tmp = _values[n];
				visited[n] = true;

				i = n;
				j = (i % _sizes[0]) * _sizes[1] + i / _sizes[0];

				while (j != n)
				{
					_values[i] = _values[j];
					visited[j] = true;

					i = j;
					j = (i % _sizes[0]) * _sizes[1] + i / _sizes[0];
				}

				_values[i] = tmp;
			}

			std::swap(_sizes[0], _sizes[1]);
		}

		return *this;
	}

	template<typename TValue>
	constexpr Matrix<TValue>& Matrix<TValue>::inverse()
	{
		assert(_sizes[0] == _sizes[1]);

		static constexpr TValue _zero = 0;
		static constexpr TValue _one = 1;

		const uint64_t size = _sizes[0];
		Matrix<ValueType> copy(*this);

		TValue* it = nullptr;
		TValue* copyIt = nullptr;
		TValue* otherIt = nullptr;
		TValue* copyOtherIt = nullptr;
		const TValue* itEnd = nullptr;
		const TValue* copyItEnd = nullptr;
		const TValue* otherItEnd = nullptr;
		const TValue* copyOtherItEnd = nullptr;

		std::fill_n(_values, _length, _zero);

		it = _values;
		itEnd = it + _length + size;
		for (; it != itEnd; it += size + 1)
		{
			*it = _one;
		}

		it = _values;
		itEnd = it + size;
		copyIt = copy._values;
		copyItEnd = copyIt + size;
		for (uint64_t j = 0; j < size; ++j, it += size + 1, itEnd += size, copyIt += size + 1, copyItEnd += size)
		{
			const TValue pivot = *copyIt;

			if (pivot == _zero)
			{
				otherIt = it + size;
				otherItEnd = itEnd + size;
				copyOtherIt = copyIt + size;
				copyOtherItEnd = copyItEnd + size;
				for (uint64_t i = j + 1; i < size; ++i, otherIt += size, otherItEnd += size, copyOtherIt += size, copyOtherItEnd += size)
				{
					const TValue value = *copyOtherIt;
					if (value != _zero)
					{
						std::transform<const TValue*>(it, itEnd, otherIt, it, [&](const TValue& x, const TValue& y) { return x + y / value; });
						std::transform<const TValue*>(copyIt, copyItEnd, copyOtherIt, copyIt, [&](const TValue& x, const TValue& y) { return x + y / value; });

						break;
					}
				}

				if (*copyIt == _zero)
				{
					throw std::runtime_error("The matrix cannot be inverted.");
				}
			}
			else if (pivot != _one)
			{
				std::transform<const TValue*>(it, itEnd, it, [&](const TValue& x) { return x / pivot; });
				std::transform<const TValue*>(copyIt, copyItEnd, copyIt, [&](const TValue& x) { return x / pivot; });
			}

			otherIt = it + size;
			otherItEnd = itEnd + size;
			copyOtherIt = copyIt + size;
			copyOtherItEnd = copyItEnd + size;
			for (uint64_t i = j + 1; i < size; ++i, otherIt += size, otherItEnd += size, copyOtherIt += size, copyOtherItEnd += size)
			{
				const TValue value = *copyOtherIt;
				if (value != _zero)
				{
					std::transform<const TValue*>(otherIt, otherItEnd, it, otherIt, [&](const TValue& x, const TValue& y) { return x - y * value; });
					std::transform<const TValue*>(copyOtherIt, copyOtherItEnd, copyIt, copyOtherIt, [&](const TValue& x, const TValue& y) { return x - y * value; });
				}
			}
		}

		it = _values + (size - 1) * size;
		itEnd = it + size;
		copyIt = copy._values + (size - 1) * size;
		copyItEnd = copyIt + size;
		for (uint64_t j = size - 1; j != UINT64_MAX; --j, it -= size, itEnd -= size, copyIt -= size, copyItEnd -= size)
		{
			otherIt = it - size;
			otherItEnd = it;
			copyOtherIt = copyIt - size;
			copyOtherItEnd = copyIt;
			for (uint64_t i = j - 1; i != UINT64_MAX; --i, otherIt -= size, otherItEnd -= size, copyOtherIt -= size, copyOtherItEnd -= size)
			{
				const TValue value = *(copyOtherIt + j);
				if (value != _zero)
				{
					std::transform<const TValue*>(otherIt, otherItEnd, it, otherIt, [&](const TValue& x, const TValue& y) { return x - y * value; });
					std::transform<const TValue*>(copyOtherIt, copyOtherItEnd, copyIt, copyOtherIt, [&](const TValue& x, const TValue& y) { return x - y * value; });
				}
			}
		}

		return *this;
	}

	template<typename TValue>
	constexpr TValue Matrix<TValue>::determinant() const
	{
		assert(_sizes[0] == _sizes[1]);

		static constexpr TValue _zero = 0;
		static constexpr TValue _one = 1;

		TValue det = _one;

		const uint64_t size = _sizes[0];
		Matrix<ValueType> copy(*this);

		TValue* it = nullptr;
		TValue* otherIt = nullptr;
		const TValue* itEnd = nullptr;
		const TValue* otherItEnd = nullptr;

		it = copy._values;
		itEnd = it + size;
		for (uint64_t j = 0; j < size; ++j, it += size + 1, itEnd += size)
		{
			const TValue pivot = *it;

			if (pivot == _zero)
			{
				otherIt = it + size;
				otherItEnd = itEnd + size;
				for (uint64_t i = j + 1; i < size; ++i, otherIt += size, otherItEnd += size)
				{
					const TValue value = *otherIt;
					if (value != _zero)
					{
						std::transform<const TValue*>(it, itEnd, otherIt, it, [&](const TValue& x, const TValue& y) { return x + y / value; });
						break;
					}
				}

				if (*it == _zero)
				{
					return _zero;
				}
			}
			else if (pivot != _one)
			{
				det /= pivot;
				std::transform<const TValue*>(it, itEnd, it, [&](const TValue& x) { return x / pivot; });
			}

			otherIt = it + size;
			otherItEnd = itEnd + size;
			for (uint64_t i = j + 1; i < size; ++i, otherIt += size, otherItEnd += size)
			{
				const TValue value = *otherIt;
				if (value != _zero)
				{
					std::transform<const TValue*>(otherIt, otherItEnd, it, otherIt, [&](const TValue& x, const TValue& y) { return x - y * value; });
				}
			}
		}

		it = copy._values;
		itEnd = it + _length + size;
		for (; it != itEnd; it += size + 1)
		{
			det *= *it;
		}

		return det;
	}


	SCP_VECTOR_DEF(template<typename TValue>, Vector, Vector<TValue>)

	template<typename TValue>
	constexpr Vector<TValue>* Vector<TValue>::createAroundMemory(uint64_t size, TValue* memory)
	{
		assert(size != 0);
		assert(memory);

		Vector<TValue>* vector = new Vector<TValue>();

		vector->_order = 2;

		vector->_length = size;

		vector->_sizes = new uint64_t[1];
		vector->_sizes[0] = size;

		vector->_values = memory;

		vector->_owner = false;

		return vector;
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB>
	constexpr void Vector<TValue>::rightMatrixProduct(const TTensorA& vector, const TTensorB& matrix) const
	{
		assert(vector.getOrder() == 1);
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == vector.getSize(0));
		assert(matrix.getSize(1) == _length);

		std::fill_n(_values, _length, 0);

		const uint64_t n = matrix.getElementCount();

		uint64_t j = 0;
		const TValue* x = &vector.get(j);

		TValue* it = _values;
		const TValue* const itEnd = it + _length;
		for (uint64_t i = 0; i < n; ++i, ++it)
		{
			if (it == itEnd)
			{
				it = _values;
				x = &vector.get(++j);
			}

			*it += *x * matrix.get(i);
		}
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensorA, TensorConcept<TValue> TTensorB>
	constexpr void Vector<TValue>::leftMatrixProduct(const TTensorA& matrix, const TTensorB& vector) const
	{
		assert(vector.getOrder() == 1);
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == _length);
		assert(matrix.getSize(1) == vector.getSize(0));

		const uint64_t m = vector.getSize(0);
		const uint64_t n = matrix.getElementCount();

		uint64_t j = 0;
		TValue* it = _values;
		*it = 0;
		for (uint64_t i = 0, j = 0; i < n; ++i, ++j)
		{
			if (j == m)
			{
				j = 0;
				++it;
				*it = 0;
			}

			*it += matrix.get(i) * vector.get(j);
		}
	}

	template<typename TValue>
	template<TensorConcept<TValue> TTensor>
	constexpr Vector<TValue>& Vector<TValue>::crossProduct(const TTensor& vector)
	{
		assert(_length == 3);
		assert(vector.getOrder() == 1);
		assert(vector.getSize(0) == 3);
		
		const TValue xA = _values[0];
		const TValue yA = _values[1];
		const TValue zA = _values[2];
		
		const TValue& xB = vector.get(0ULL);
		const TValue& yB = vector.get(1ULL);
		const TValue& zB = vector.get(2ULL);
		
		_values[0] = yA * zB - zA * yB;
		_values[1] = zA * xB - xA * zB;
		_values[2] = xA * yB - yA * xB;
		
		return *this;
	}
}
