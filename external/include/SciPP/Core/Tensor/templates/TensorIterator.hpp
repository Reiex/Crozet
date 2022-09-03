#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/TensorIterator.hpp>

namespace scp
{
	constexpr DynTensorIterator::DynTensorIterator(const TensorShaped* tensorShaped, bool end) : TensorIteratorBase(tensorShaped, new uint64_t[tensorShaped->getOrder()], end)
	{
	}

	constexpr DynTensorIterator::DynTensorIterator(const DynTensorIterator& iterator) : TensorIteratorBase(iterator, new uint64_t[iterator._tensorShaped->getOrder()])
	{
	}

	constexpr DynTensorIterator::DynTensorIterator(DynTensorIterator&& iterator) : TensorIteratorBase(std::move(iterator), iterator._position.indices)
	{
	}

	constexpr DynTensorIterator& DynTensorIterator::operator=(const DynTensorIterator& iterator)
	{
		const uint64_t iteratorOrder = iterator._tensorShaped->getOrder();

		if (_tensorShaped->getOrder() != iteratorOrder)
		{
			delete _position.indices;
			copyFrom(iterator, new uint64_t[iteratorOrder]);
		}
		else
		{
			copyFrom(iterator, _position.indices);
		}

		return *this;
	}

	constexpr DynTensorIterator& DynTensorIterator::operator=(DynTensorIterator&& iterator)
	{
		delete _position.indices;

		moveFrom(std::move(iterator), iterator._position.indices);

		return *this;
	}

	constexpr DynTensorIterator::~DynTensorIterator()
	{
		if (_position.indices)
		{
			delete _position.indices;
		}
	}


	template<uint64_t Order>
	constexpr TensorIterator<Order>::TensorIterator(const TensorShaped* tensorShaped, bool end) : TensorIteratorBase(tensorShaped, _indices, end)
	{
		assert(_tensorShaped->getOrder() == Order);
	}

	template<uint64_t Order>
	constexpr TensorIterator<Order>::TensorIterator(const TensorIterator<Order>& iterator) : TensorIteratorBase(iterator, _indices)
	{
	}

	template<uint64_t Order>
	constexpr TensorIterator<Order>::TensorIterator(TensorIterator<Order>&& iterator) : TensorIteratorBase(std::move(iterator), _indices)
	{
		std::copy(iterator._indices, iterator._indices + Order, _indices);
	}

	template<uint64_t Order>
	constexpr TensorIterator<Order>& TensorIterator<Order>::operator=(const TensorIterator<Order>& iterator)
	{
		copyFrom(iterator, _indices);

		return *this;
	}

	template<uint64_t Order>
	constexpr TensorIterator<Order>& TensorIterator<Order>::operator=(TensorIterator<Order>&& iterator)
	{
		moveFrom(std::move(iterator), _indices);

		std::copy(iterator._indices, iterator._indices + Order, _indices);

		return *this;
	}


	constexpr DynTensorIterator begin(const TensorShaped& tensorShaped)
	{
		return DynTensorIterator(&tensorShaped, false);
	}

	constexpr DynTensorIterator end(const TensorShaped& tensorShaped)
	{
		return DynTensorIterator(&tensorShaped, true);
	}
}
