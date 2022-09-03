#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/TensorShaped.hpp>

namespace scp
{
	constexpr TensorIteratorBase::TensorIteratorBase(const TensorShaped* tensorShaped, uint64_t* indicesPtr, bool end) :
		_tensorShaped(tensorShaped),
		_position{ .indices = indicesPtr, .internalIndex = UINT64_MAX },
		_order(tensorShaped->getOrder()),
		_shape(tensorShaped->getShape()),
		_internalCounter(0),
		_savedIndex(UINT64_MAX)
	{
		std::fill(_position.indices, _position.indices + _order, 0);
		if (end)
		{
			_position.indices[0] = _tensorShaped->getSize(0);
			_internalCounter = _tensorShaped->getTotalLength();
		}

		_tensorShaped->setInitialPosition(this, end);
	}

	constexpr TensorIteratorBase::TensorIteratorBase(const TensorIteratorBase& iterator, uint64_t* indicesPtr)
	{
		copyFrom(iterator, indicesPtr);
	}

	constexpr TensorIteratorBase::TensorIteratorBase(TensorIteratorBase&& iterator, uint64_t* indicesPtr)
	{
		moveFrom(std::move(iterator), indicesPtr);
	}

	constexpr const TensorPosition& TensorIteratorBase::operator*() const
	{
		return _position;
	}

	constexpr const TensorPosition* TensorIteratorBase::operator->() const
	{
		return &_position;
	}

	constexpr TensorIteratorBase& TensorIteratorBase::operator++()
	{
		uint64_t i = _order - 1;
		for (; _position.indices[i] == _shape[i] - 1 && i != 0; --i)
		{
			_position.indices[i] = 0;
		}

		++_position.indices[i];
		++_internalCounter;

		_tensorShaped->incrementPosition(this);

		return *this;
	}

	constexpr bool TensorIteratorBase::operator==(const TensorIteratorBase& iterator) const
	{
		return _internalCounter == iterator._internalCounter && _order == iterator._order && std::equal(_shape, _shape + _order, iterator._shape);
	}

	constexpr bool TensorIteratorBase::operator!=(const TensorIteratorBase& iterator) const
	{
		return !(*this == iterator);
	}

	constexpr const uint64_t* TensorIteratorBase::getIndices() const
	{
		return _position.indices;
	}

	constexpr uint64_t* TensorIteratorBase::getIndices()
	{
		return _position.indices;
	}

	constexpr const uint64_t& TensorIteratorBase::getInternalIndex() const
	{
		return _position.internalIndex;
	}

	constexpr uint64_t& TensorIteratorBase::getInternalIndex()
	{
		return _position.internalIndex;
	}

	constexpr  const uint64_t& TensorIteratorBase::getSavedIndex() const
	{
		return _savedIndex;
	}

	constexpr uint64_t& TensorIteratorBase::getSavedIndex()
	{
		return _savedIndex;
	}

	constexpr void TensorIteratorBase::copyFrom(const TensorIteratorBase& iterator, uint64_t* indicesPtr)
	{
		_tensorShaped = iterator._tensorShaped;
		_position.indices = indicesPtr;
		_position.internalIndex = iterator._position.internalIndex;
		_order = iterator._order;
		_shape = iterator._shape;
		_internalCounter = iterator._internalCounter;
		_savedIndex = iterator._savedIndex;

		std::copy(iterator._position.indices, iterator._position.indices + _order, indicesPtr);
	}

	constexpr void TensorIteratorBase::moveFrom(TensorIteratorBase&& iterator, uint64_t* indicesPtr)
	{
		_tensorShaped = iterator._tensorShaped;
		_position.indices = indicesPtr;
		_position.internalIndex = iterator._position.internalIndex;
		_order = iterator._order;
		_shape = iterator._shape;
		_internalCounter = iterator._internalCounter;
		_savedIndex = iterator._savedIndex;

		iterator._tensorShaped = nullptr;
		iterator._position.indices = nullptr;
		iterator._position.internalIndex = UINT64_MAX;
		iterator._order = 0;
		iterator._shape = nullptr;
		iterator._internalCounter = 0;
		iterator._savedIndex = UINT64_MAX;
	}

	constexpr uint64_t TensorShaped::getTotalLength() const
	{
		const uint64_t* const it = getShape();
		return std::accumulate(it, it + getOrder(), 1, std::multiplies<uint64_t>());
	}
}
