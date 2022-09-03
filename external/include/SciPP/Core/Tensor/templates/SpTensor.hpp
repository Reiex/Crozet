#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/SpTensor.hpp>

namespace scp
{
	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement::SpTensorElement(uint64_t eltOrder, const uint64_t* eltIndices, const TValue& eltValue) :
		order(eltOrder),
		indices(new uint64_t[eltOrder]),
		value(eltValue)
	{
		std::copy(eltIndices, eltIndices + eltOrder, indices);
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement::SpTensorElement(const SpTensor<TValue>::SpTensorElement& elt) :
		order(elt.order),
		indices(new uint64_t[elt.order]),
		value(elt.value)
	{
		std::copy(elt.indices, elt.indices + elt.order, indices);
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement::SpTensorElement(SpTensor<TValue>::SpTensorElement&& elt) :
		order(elt.order),
		indices(elt.indices),
		value(std::move(elt.value))
	{
		elt.order = 0;
		elt.indices = nullptr;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement& SpTensor<TValue>::SpTensorElement::operator=(const SpTensor<TValue>::SpTensorElement& elt)
	{
		if (order != elt.order)
		{
			delete[] indices;

			order = elt.order;
			indices = new uint64_t[elt.order];
		}

		std::copy(elt.indices, elt.indices + elt.order, indices);
		value = elt.value;

		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement& SpTensor<TValue>::SpTensorElement::operator=(SpTensor<TValue>::SpTensorElement&& elt)
	{
		delete[] indices;

		order = elt.order;
		indices = elt.indices;
		value = std::move(elt.value);

		elt.order = 0;
		elt.indices = nullptr;

		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensorElement::~SpTensorElement()
	{
		if (indices)
		{
			delete[] indices;
		}
	}


	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor() :
		_order(0),
		_shape(nullptr),
		_values()
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(uint64_t order, const uint64_t* shape) : SpTensor<TValue>()
	{
		create(order, shape);
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(uint64_t order, const uint64_t* shape, const TValue& value) : SpTensor<TValue>(order, shape)
	{
		if (value != _zero)
		{
			for (const TensorPosition& pos : *this)
			{
				_values.emplace_back(order, pos.indices, value);
			}
		}
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(uint64_t order, const uint64_t* shape, const TValue* values) : SpTensor<TValue>(order, shape)
	{
		const TValue* it = values;
		for (const TensorPosition& pos : *this)
		{
			if (*it != _zero)
			{
				_values.emplace_back(order, pos.indices, *it);
			}

			++it;
		}
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(uint64_t order, const uint64_t* shape, const std::vector<TValue>& values) : SpTensor<TValue>(order, shape, values.data())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(uint64_t order, const uint64_t* shape, const std::initializer_list<TValue>& values) : SpTensor<TValue>(order, shape, values.begin())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::vector<uint64_t>& shape) : SpTensor<TValue>(shape.size(), shape.data())
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::vector<uint64_t>& shape, const TValue& value) : SpTensor<TValue>(shape.size(), shape.data(), value)
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::vector<uint64_t>& shape, const TValue* values) : SpTensor<TValue>(shape.size(), shape.data(), values)
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::vector<uint64_t>& shape, const std::vector<TValue>& values) : SpTensor<TValue>(shape.size(), shape.data(), values.data())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::vector<uint64_t>& shape, const std::initializer_list<TValue>& values) : SpTensor<TValue>(shape.size(), shape.data(), values.begin())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::initializer_list<uint64_t>& shape) : SpTensor<TValue>(shape.size(), shape.begin())
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::initializer_list<uint64_t>& shape, const TValue& value) : SpTensor<TValue>(shape.size(), shape.begin(), value)
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::initializer_list<uint64_t>& shape, const TValue* values) : SpTensor<TValue>(shape.size(), shape.begin(), values)
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::initializer_list<uint64_t>& shape, const std::vector<TValue>& values) : SpTensor<TValue>(shape.size(), shape.begin(), values.data())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const std::initializer_list<uint64_t>& shape, const std::initializer_list<TValue>& values) : SpTensor<TValue>(shape.size(), shape.begin(), values.begin())
	{
		assert(values.size() == this->getTotalLength());
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const TensorBase<TValue>& tensor) : SpTensor<TValue>()
	{
		create(tensor.getOrder(), tensor.getShape());
		copyFrom(tensor);
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(const SpTensor<TValue>& tensor) : SpTensor<TValue>(dynamic_cast<const TensorBase<TValue>&>(tensor))
	{
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::SpTensor(SpTensor<TValue>&& tensor) : SpTensor<TValue>()
	{
		moveFrom(std::move(tensor));
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator=(const SpTensor<TValue>& tensor)
	{
		copyFrom(tensor);
		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator=(SpTensor<TValue>&& tensor)
	{
		moveFrom(tensor);
		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>* SpTensor<TValue>::clone() const
	{
		return new SpTensor<TValue>(*this);
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator+=(const TensorBase<TValue>& tensor)
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			assert(_order == spTensor->_order);
			assert(std::equal(_shape, _shape + _order, spTensor->_shape));

			if (_values.empty())
			{
				_values = spTensor->_values;
			}
			else if (!spTensor->_values.empty())
			{
				auto it = _values.begin();
				auto spIt = spTensor->_values.begin();
				const auto spItEnd = spTensor->_values.end();

				for (; spIt != spItEnd; ++spIt)
				{
					while (it != _values.end() && std::lexicographical_compare(it->indices, it->indices + _order, spIt->indices, spIt->indices + _order))
					{
						++it;
					}

					if (it == _values.end())
					{
						_values.insert(it, spIt, spItEnd);
						break;
					}
					else if (std::equal(it->indices, it->indices + _order, spIt->indices))
					{
						it->value += spIt->value;
						if (it->value == _zero)
						{
							it = _values.erase(it);
						}
						else
						{
							++it;
						}
					}
					else
					{
						it = _values.emplace(it, _order, spIt->indices, spIt->value);
						++it;
					}
				}
			}
		}
		else
		{
			TensorBase<TValue>::operator+=(tensor);
		}

		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator-=(const TensorBase<TValue>& tensor)
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			assert(_order == spTensor->_order);
			assert(std::equal(_shape, _shape + _order, spTensor->_shape));

			if (_values.empty())
			{
				for (const SpTensorElement& elt : spTensor->_values)
				{
					_values.emplace_back(_order, elt.indices, -elt.value);
				}
			}
			else if (!spTensor->_values.empty())
			{
				auto it = _values.begin();
				auto spIt = spTensor->_values.begin();
				const auto spItEnd = spTensor->_values.end();

				for (; spIt != spItEnd; ++spIt)
				{
					while (it != _values.end() && std::lexicographical_compare(it->indices, it->indices + _order, spIt->indices, spIt->indices + _order))
					{
						++it;
					}

					if (it == _values.end())
					{
						for (; spIt != spItEnd; ++spIt)
						{
							_values.emplace_back(_order, spIt->indices, -spIt->value);
						}
					}
					else if (std::equal(it->indices, it->indices + _order, spIt->indices))
					{
						it->value -= spIt->value;
						if (it->value == _zero)
						{
							it = _values.erase(it);
						}
						else
						{
							++it;
						}
					}
					else
					{
						it = _values.emplace(it, _order, spIt->indices, -spIt->value);
						++it;
					}
				}
			}
		}
		else
		{
			TensorBase<TValue>::operator-=(tensor);
		}

		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator*=(const TValue& value)
	{
		if (value == _zero)
		{
			_values.clear();
		}
		else
		{
			for (SpTensorElement& elt : _values)
			{
				elt.value *= value;
			}
		}

		return *this;
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::operator/=(const TValue& value)
	{
		for (SpTensorElement& elt : _values)
		{
			elt.value /= value;
		}

		return *this;
	}

	template<typename TValue>
	constexpr bool SpTensor<TValue>::operator==(const TensorBase<TValue>& tensor) const
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			if (_order != spTensor->_order)
			{
				return false;
			}

			if (!std::equal(_shape, _shape + _order, spTensor->_shape))
			{
				return false;
			}

			auto it = _values.begin();
			const auto itEnd = _values.end();
			auto spIt = spTensor->_values.begin();
			for (; it != itEnd; ++it, ++spIt)
			{
				if (it->value != spIt->value)
				{
					return false;
				}

				if (!std::equal(it->indices, it->indices + _order, spIt->indices))
				{
					return false;
				}
			}
		}
		else
		{
			return TensorBase<TValue>::operator==(tensor);
		}
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::hadamardProduct(const TensorBase<TValue>& tensor)
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			assert(_order == spTensor->_order);
			assert(std::equal(_shape, _shape + _order, spTensor->_shape));

			if (spTensor->_values.empty())
			{
				_values.clear();
			}
			else if (!_values.empty())
			{
				auto it = _values.begin();
				auto spIt = spTensor->_values.begin();
				const auto spItEnd = spTensor->_values.end();

				for (; spIt != spItEnd; ++spIt)
				{
					while (it != _values.end() && std::lexicographical_compare(it->indices, it->indices + _order, spIt->indices, spIt->indices + _order))
					{
						++it;
					}

					if (it == _values.end())
					{
						break;
					}
					else if (std::equal(it->indices, it->indices + _order, spIt->indices))
					{
						it->value *= spIt->value;
						if (it->value == _zero)
						{
							it = _values.erase(it);
						}
						else
						{
							++it;
						}
					}
					else
					{
						++it;
					}
				}
			}

			return *this;
		}
		else
		{
			return dynamic_cast<SpTensor<TValue>&>(TensorBase<TValue>::hadamardProduct(tensor));
		}
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::convolution(const TensorBase<TValue>& kernel, ConvolutionMethod method)
	{
		const SpTensor<TValue>* spKernel = dynamic_cast<const SpTensor<TValue>*>(&kernel);
		if (spKernel)
		{
			assert(_order == spKernel->_order);

			// Make a copy of *this and set *this to 0
			SpTensor<TValue> copy(*this);
			_values.clear();

			// Check that the kernel's shape is odd
			for (uint64_t i = 0; i < _order; i++)
			{
				assert(spKernel->_shape[i] % 2 == 1);
				assert(spKernel->_shape[i] <= _shape[i]);
			}

			// Compute offset (to center the kernel)
			int64_t* offset = reinterpret_cast<int64_t*>(alloca(_order * sizeof(int64_t)));
			for (uint64_t i = 0; i < _order; i++)
			{
				offset[i] = static_cast<int64_t>(spKernel->_shape[i] / 2);
			}

			int64_t* offsetedIndices = reinterpret_cast<int64_t*>(alloca(_order * sizeof(int64_t)));

			// For each non zero element of the original tensor
			for (const SpTensorElement& elt : copy._values)
			{
				// For each non zero element of the kernel
				for (const SpTensorElement& kernelElt : spKernel->_values)
				{
					bool setToZero = false;

					int64_t* itOffsetedIndices = offsetedIndices;
					const uint64_t* itShape = _shape;
					const int64_t* itOffset = offset;
					const uint64_t* itIndices = elt.indices;
					const uint64_t* itKernelIndices = kernelElt.indices;

					// Compute the corresponding indices in the result
					for (uint64_t k = 0; k < _order; ++k, ++itOffsetedIndices, ++itShape, ++itOffset, ++itIndices, ++itKernelIndices)
					{
						*itOffsetedIndices = static_cast<int64_t>(*itIndices) + static_cast<int64_t>(*itKernelIndices) - *itOffset;

						switch (method)
						{
						case ConvolutionMethod::Zero:
							setToZero = (*itOffsetedIndices < 0 || *itOffsetedIndices >= *itShape);
							break;
						case ConvolutionMethod::Continuous:
							*itOffsetedIndices = std::clamp<int64_t>(*itOffsetedIndices, 0, *itShape - 1);
							break;
						case ConvolutionMethod::Periodic:
							*itOffsetedIndices = (*itOffsetedIndices + *itShape) % *itShape;
							break;
						default:
							assert(false);
						}

						if (setToZero)
						{
							break;
						}
					}

					// Add the product to the result
					if (!setToZero)
					{
						bool found;
						const uint64_t sparseIndex = findSpElement(reinterpret_cast<uint64_t*>(offsetedIndices), found);
						if (found)
						{
							_values[sparseIndex].value += elt.value * kernelElt.value;
						}
						else
						{
							_values.emplace(_values.begin() + sparseIndex, _order, reinterpret_cast<uint64_t*>(offsetedIndices), elt.value * kernelElt.value);
						}
					}
				}
			}

			return *this;
		}
		else
		{
			// TODO : Could be done even with kernel non-sparse
			return dynamic_cast<SpTensor<TValue>&>(TensorBase<TValue>::convolution(kernel, method));
		}
	}

	template<typename TValue>
	constexpr SpTensor<TValue>& SpTensor<TValue>::negate()
	{
		for (SpTensorElement& elt : _values)
		{
			elt.value = -elt.value;
		}

		return *this;
	}

	template<typename TValue>
	constexpr TValue SpTensor<TValue>::dotProduct(const TensorBase<TValue>& tensor) const
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			assert(_order == spTensor->_order);
			assert(std::equal(_shape, _shape + _order, spTensor->_shape));

			TValue result = _zero;

			if (!_values.empty() && !spTensor->_values.empty())
			{
				auto it = _values.begin();
				auto spIt = spTensor->_values.begin();
				const auto spItEnd = spTensor->_values.end();

				for (; spIt != spItEnd; ++spIt, ++it)
				{
					while (it != _values.end() && std::lexicographical_compare(it->indices, it->indices + _order, spIt->indices, spIt->indices + _order))
					{
						++it;
					}

					if (it == _values.end())
					{
						break;
					}
					else if (std::equal(it->indices, it->indices + _order, spIt->indices))
					{
						result += it->value * spIt->value;
					}
				}
			}

			return result;
		}
		else
		{
			return TensorBase<TValue>::dotProduct(tensor);
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		SpTensor<TValue>* spResult = dynamic_cast<SpTensor<TValue>*>(&result);
		if (spTensor && spResult) // TODO
		{
			assert(_order + spTensor->_order == spResult->_order);
			assert(std::equal(_shape, _shape + _order, spResult->_shape) && std::equal(spTensor->_shape, spTensor->_shape + spTensor->_order, spResult->_shape + _order));

			spResult->_values.clear();

			uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(spResult->_order * sizeof(uint64_t)));
			for (const SpTensorElement& elt : _values)
			{
				for (const SpTensorElement& spElt : spTensor->_values)
				{
					std::copy(elt.indices, elt.indices + _order, indices);
					std::copy(spElt.indices, spElt.indices + spTensor->_order, indices + _order);
					spResult->_values.emplace_back(spResult->_order, indices, elt.value * spElt.value);
				}
			}
		}
		else
		{
			TensorBase<TValue>::tensorProduct(tensor, result);
		}
	}

	template<typename TValue>
	constexpr TValue SpTensor<TValue>::normSq() const
	{
		TValue result = _zero;
		for (const SpTensorElement& elt : _values)
		{
			result += elt.value * elt.value;
		}

		return result;
	}

	template<typename TValue>
	constexpr const TValue& SpTensor<TValue>::minElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;

		if constexpr (isTotallyOrdered)
		{
			if (_values.empty())
			{
				return _zero;
			}

			auto minIt = _values.begin();
			auto it = minIt + 1;
			const auto itEnd = _values.end();
			for (; it != itEnd; ++it)
			{
				if (it->value < minIt->value)
				{
					minIt = it;
				}
			}

			if (minIt->value > 0)
			{
				if (_values.size() < this->getTotalLength())
				{
					return _zero;
				}
			}

			return minIt->value;
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& SpTensor<TValue>::maxElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;

		if constexpr (isTotallyOrdered)
		{
			if (_values.empty())
			{
				return _zero;
			}

			auto maxIt = _values.begin();
			auto it = maxIt + 1;
			const auto itEnd = _values.end();
			for (; it != itEnd; ++it)
			{
				if (it->value > maxIt->value)
				{
					maxIt = it;
				}
			}

			if (maxIt->value < 0)
			{
				if (_values.size() < this->getTotalLength())
				{
					return _zero;
				}
			}

			return maxIt->value;
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& SpTensor<TValue>::get(uint64_t internalIndex) const
	{
		if (internalIndex < _values.size())
		{
			return _values[internalIndex].value;
		}
		else
		{
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& SpTensor<TValue>::get(const uint64_t* indices) const
	{
		bool found;
		uint64_t sparseIndex = findSpElement(indices, found);
		if (found)
		{
			return _values[sparseIndex].value;
		}
		else
		{
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& SpTensor<TValue>::get(const std::initializer_list<uint64_t>& indices) const
	{
		assert(indices.size() == _order);

		bool found;
		uint64_t sparseIndex = findSpElement(indices.begin(), found);
		if (found)
		{
			return _values[sparseIndex].value;
		}
		else
		{
			return _zero;
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::set(uint64_t internalIndex, const TValue& value)
	{
		assert(internalIndex < _values.size());
		if (value == _zero)
		{
			_values.erase(_values.begin() + internalIndex);
		}
		else
		{
			_values[internalIndex].value = value;
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::set(const uint64_t* indices, const TValue& value)
	{
		bool found;
		uint64_t sparseIndex = findSpElement(indices, found);
		if (found)
		{
			_values[sparseIndex].value = value;
		}
		else
		{
			_values.emplace(_values.begin() + sparseIndex, _order, indices, value);
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::set(const std::initializer_list<uint64_t>& indices, const TValue& value)
	{
		assert(indices.size() == _order);

		bool found;
		uint64_t sparseIndex = findSpElement(indices.begin(), found);
		if (found)
		{
			_values[sparseIndex].value = value;
		}
		else
		{
			_values.emplace(_values.begin() + sparseIndex, _order, indices.begin(), value);
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::getIndices(uint64_t internalIndex, uint64_t* indices) const
	{
		assert(internalIndex < _values.size());
		std::copy(_values[internalIndex].indices, _values[internalIndex].indices + _order, indices);
	}

	template<typename TValue>
	constexpr uint64_t SpTensor<TValue>::getInternalIndex(const uint64_t* indices) const
	{
		bool found;
		uint64_t sparseIndex = findSpElement(indices, found);
		if (found)
		{
			return sparseIndex;
		}
		else
		{
			return -1;
		}
	}

	template<typename TValue>
	constexpr uint64_t SpTensor<TValue>::getInternalLength() const
	{
		return _values.size();
	}

	template<typename TValue>
	constexpr uint64_t SpTensor<TValue>::getOrder() const
	{
		return _order;
	}

	template<typename TValue>
	constexpr const uint64_t* SpTensor<TValue>::getShape() const
	{
		return _shape;
	}

	template<typename TValue>
	constexpr uint64_t SpTensor<TValue>::getSize(uint64_t i) const
	{
		return _shape[i];
	}

	template<typename TValue>
	constexpr SpTensor<TValue>::~SpTensor()
	{
		destroy();
	}

	template<typename TValue>
	constexpr uint64_t SpTensor<TValue>::findSpElement(const uint64_t* indices, bool& found) const
	{
		for (uint64_t i = 0; i < _order; ++i)
		{
			assert(indices[i] < _shape[i]);
		}

		if (_values.empty())
		{
			found = false;
			return 0;
		}

		uint64_t a = 0, b = _values.size() - 1, m;
		if (!std::lexicographical_compare(_values[a].indices, _values[a].indices + _order, indices, indices + _order))
		{
			found = std::equal(indices, indices + _order, _values[a].indices);
			return a;
		}

		if (!std::lexicographical_compare(indices, indices + _order, _values[b].indices, _values[b].indices + _order))
		{
			found = std::equal(indices, indices + _order, _values[b].indices);
			if (found)
			{
				return b;
			}
			else
			{
				return b + 1;
			}
		}

		while (b - a > 1)
		{
			m = (a + b) / 2;
			if (std::lexicographical_compare(_values[m].indices, _values[m].indices + _order, indices, indices + _order))
			{
				a = m;
			}
			else if (std::lexicographical_compare(indices, indices + _order, _values[m].indices, _values[m].indices + _order))
			{
				b = m;
			}
			else
			{
				found = true;
				return m;
			}
		}

		found = false;
		return b;
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::setInitialPosition(TensorIteratorBase* iterator, bool end) const
	{
		bool isFirstPos = true;
		if (!_values.empty())
		{
			const uint64_t* it = _values.front().indices;
			const uint64_t* const itEnd = _values.front().indices + _order;
			for (; it != itEnd; ++it)
			{
				if (*it != 0)
				{
					isFirstPos = false;
					break;
				}
			}
		}

		if (isFirstPos)
		{
			iterator->getSavedIndex() = 0;
			iterator->getInternalIndex() = 0;
		}
		else
		{
			iterator->getSavedIndex() = UINT64_MAX;
			iterator->getInternalIndex() = UINT64_MAX;
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::incrementPosition(TensorIteratorBase* iterator) const
	{
		// If iterator is past the last element
		if (iterator->getSavedIndex() == _values.size())
		{
			return;
		}

		// Compute the index to evaluate
		uint64_t index = 0;
		if (iterator->getSavedIndex() != UINT64_MAX)
		{
			index = iterator->getSavedIndex() + 1;
		}

		// If the index is past the last element, we just got past the last element
		if (index == _values.size())
		{
			iterator->getInternalIndex() = UINT64_MAX;
			iterator->getSavedIndex() = index;
			return;
		}

		// Advance normally
		if (std::equal(_values[index].indices, _values[index].indices + _order, iterator->getIndices()))
		{
			iterator->getInternalIndex() = index;
			iterator->getSavedIndex() = index;
		}
		else
		{
			iterator->getInternalIndex() = UINT64_MAX;
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::create(uint64_t order, const uint64_t* shape)
	{
		assert(order != 0);
		assert(_values.size() == 0);

		_order = order;
		_shape = new uint64_t[order];
		std::copy(shape, shape + order, _shape);
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::copyFrom(const TensorBase<TValue>& tensor)
	{
		const SpTensor<TValue>* spTensor = dynamic_cast<const SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			if (_order != spTensor->_order)
			{
				destroy();
				create(spTensor->_order, spTensor->_shape);
			}

			_values = spTensor->_values;
		}
		else
		{
			TensorBase<TValue>::copyFrom(tensor);
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::moveFrom(TensorBase<TValue>&& tensor)
	{
		SpTensor<TValue>* spTensor = dynamic_cast<SpTensor<TValue>*>(&tensor);
		if (spTensor)
		{
			destroy();

			_order = spTensor->_order;
			_shape = spTensor->_shape;
			_values = std::move(spTensor->_values);

			spTensor->_order = 0;
			spTensor->_shape = nullptr;
		}
		else
		{
			TensorBase<TValue>::moveFrom(std::move(tensor));
		}
	}

	template<typename TValue>
	constexpr void SpTensor<TValue>::destroy()
	{
		if (_shape)
		{
			delete[] _shape;

			_order = 0;
			_shape = nullptr;
			_values.clear();
		}
	}


	template<typename TValue>
	constexpr DynTensorIterator begin(const SpTensor<TValue>& tensor)
	{
		return DynTensorIterator(&tensor, false);
	}

	template<typename TValue>
	constexpr DynTensorIterator end(const SpTensor<TValue>& tensor)
	{
		return DynTensorIterator(&tensor, true);
	}


	template<typename TValue>
	SpTensor<TValue> tensorContraction(const SpTensor<TValue>& tensor, uint64_t i, uint64_t j)
	{
		assert(tensor.getOrder() > 2);
		assert(i != j);
		assert(tensor.getSize(i) == tensor.getSize(j));

		if (i > j)
		{
			std::swap(i, j);
		}

		const uint64_t tensorOrder = tensor.getOrder();
		const uint64_t* tensorShape = tensor.getShape();
		uint64_t* shape = reinterpret_cast<uint64_t*>(alloca((tensorOrder - 2) * sizeof(uint64_t)));
		if (i != 0)
		{
			std::copy(tensorShape, tensorShape + i, shape);
		}
		if (i != j - 1)
		{
			std::copy(tensorShape + i + 1, tensorShape + j, shape + i);
		}
		if (j != tensorOrder - 1)
		{
			std::copy(tensorShape + j + 1, tensorShape + tensorOrder, shape + j - 1);
		}

		SpTensor<TValue> contraction(tensorOrder - 2, shape);
		tensor.tensorContraction(i, j, contraction);

		return contraction;
	}

	template<typename TValue>
	SpTensor<TValue> tensorProduct(const SpTensor<TValue>& a, const SpTensor<TValue>& b)
	{
		const uint64_t orderA = a.getOrder();
		const uint64_t orderB = b.getOrder();
		const uint64_t* shapeA = a.getShape();
		const uint64_t* shapeB = b.getShape();

		uint64_t* shape = reinterpret_cast<uint64_t*>(alloca((orderA + orderB) * sizeof(uint64_t)));
		std::copy(shapeA, shapeA + orderA, shape);
		std::copy(shapeB, shapeB + orderB, shape + orderA);

		SpTensor<TValue> c(orderA + orderB, shape);
		a.tensorProduct(b, c);

		return c;
	}

	template<typename TValue>
	SpTensor<TValue> contractedTensorProduct(const SpTensor<TValue>& a, const SpTensor<TValue>& b)
	{
		const uint64_t orderA = a.getOrder();
		const uint64_t orderB = b.getOrder();
		const uint64_t* shapeA = a.getShape();
		const uint64_t* shapeB = b.getShape();

		assert(orderA + orderB > 2);
		assert(shapeA[orderA - 1] == shapeB[0]);

		uint64_t* shape = reinterpret_cast<uint64_t*>(alloca((orderA + orderB - 2) * sizeof(uint64_t)));
		std::copy(shapeA, shapeA + orderA - 1, shape);
		std::copy(shapeB, shapeB + orderB - 1, shape + orderA - 1);

		SpTensor<TValue> c(orderA + orderB - 2, shape);
		a.contractedTensorProduct(b, c);

		return c;
	}


	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(uint64_t row, uint64_t col) : MatrixBase<SpTensor<TValue>>(std::vector<uint64_t>{ { row, col } })
	{
	}

	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(uint64_t row, uint64_t col, const TValue& value) : MatrixBase<SpTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, value)
	{
	}

	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(uint64_t row, uint64_t col, const TValue* values) : MatrixBase<SpTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(uint64_t row, uint64_t col, const std::vector<TValue>& values) : MatrixBase<SpTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(uint64_t row, uint64_t col, const std::initializer_list<TValue>& values) : MatrixBase<SpTensor<TValue>>(std::vector<uint64_t>{ { row, col } }, values)
	{
	}

	template<typename TValue>
	constexpr SpMatrix<TValue>::SpMatrix(const TensorBase<TValue>& tensor) : MatrixBase<SpTensor<TValue>>(tensor)
	{
	}


	template<typename TValue>
	SpMatrix<TValue> operator*(const SpMatrix<TValue>& a, const SpMatrix<TValue>& b)
	{
		SpMatrix<TValue> result(a.getSize(0), b.getSize(1));
		a.matrixProduct(b, result);
		return result;
	}

	template<typename TValue>
	Vector<TValue> operator*(const SpMatrix<TValue>& matrix, const Vector<TValue>& vector)
	{
		Vector<TValue> result(matrix.getSize(0));
		matrix.vectorProduct(vector, result);
		return result;
	}

	template<typename TValue>
	SpMatrix<TValue> transpose(const SpMatrix<TValue>& matrix)
	{
		SpMatrix<TValue> result(matrix.getSize(1), matrix.getSize(0));
		matrix.transpose(result);
		return result;
	}


	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(uint64_t size) : VectorBase<SpTensor<TValue>>(1, &size)
	{
	}

	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(uint64_t size, const TValue& value) : VectorBase<SpTensor<TValue>>(1, &size, value)
	{
	}

	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(uint64_t size, const TValue* values) : VectorBase<SpTensor<TValue>>(1, &size, values)
	{
	}

	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(const std::vector<TValue>& values) : VectorBase<SpTensor<TValue>>(std::vector<uint64_t>{ { values.size() } }, values.data())
	{
	}

	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(const std::initializer_list<TValue>& values) : VectorBase<SpTensor<TValue>>(std::vector<uint64_t>{ { values.size() } }, values.begin())
	{
	}

	template<typename TValue>
	constexpr SpVector<TValue>::SpVector(const TensorBase<TValue>& tensor) : VectorBase<SpTensor<TValue>>(tensor)
	{
	}


	template<typename TValue>
	SpVector<TValue> operator*(const SpVector<TValue>& vector, const SpMatrix<TValue>& matrix)
	{
		SpVector<TValue> result(matrix.getSize(1));
		vector.matrixProduct(matrix, result);
		return result;
	}
}
