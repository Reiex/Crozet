#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/TensorSweep.hpp>

namespace scp
{
	constexpr void TensorSweepBase::getIndices(uint64_t internalIndex, uint64_t* indices) const
	{
		assert(false);
	}

	constexpr uint64_t TensorSweepBase::getInternalIndex(const uint64_t* indices) const
	{
		assert(false);
		return UINT64_MAX;
	}

	constexpr uint64_t TensorSweepBase::getInternalLength() const
	{
		assert(false);
		return 0;
	}

	constexpr void TensorSweepBase::setInitialPosition(TensorIteratorBase* iterator, bool end) const
	{
	}

	constexpr void TensorSweepBase::incrementPosition(TensorIteratorBase* iterator) const
	{
	}


	constexpr DynTensorSweep::DynTensorSweep(uint64_t order, const uint64_t* shape) :
		_order(order),
		_shape(new uint64_t[order]),
		_length(std::accumulate(shape, shape + order, 1, std::multiplies<uint64_t>()))
	{
		std::copy(shape, shape + order, _shape);
	}

	constexpr DynTensorSweep::DynTensorSweep(const std::vector<uint64_t>& shape) : DynTensorSweep(shape.size(), shape.data())
	{
	}

	constexpr DynTensorSweep::DynTensorSweep(const std::initializer_list<uint64_t>& shape) : DynTensorSweep(shape.size(), shape.begin())
	{
	}

	constexpr DynTensorSweep::DynTensorSweep(const TensorShaped& tensorShaped) : DynTensorSweep(tensorShaped.getOrder(), tensorShaped.getShape())
	{
	}

	constexpr DynTensorSweep::DynTensorSweep(const DynTensorSweep& sweep) : DynTensorSweep(sweep._order, sweep._shape)
	{
	}

	constexpr DynTensorSweep::DynTensorSweep(DynTensorSweep&& sweep) :
		_order(sweep._order),
		_shape(sweep._shape),
		_length(sweep._length)
	{
		sweep._order = 0;
		sweep._shape = nullptr;
		sweep._length = 0;
	}

	constexpr DynTensorSweep& DynTensorSweep::operator=(const DynTensorSweep& sweep)
	{
		if (_order != sweep._order)
		{
			delete[] _shape;
			_order = sweep._order;
			_shape = new uint64_t[_order];
		}

		std::copy(sweep._shape, sweep._shape + _order, _shape);
		_length = sweep._length;

		return *this;
	}

	constexpr DynTensorSweep& DynTensorSweep::operator=(DynTensorSweep&& sweep)
	{
		delete[] _shape;

		_order = sweep._order;
		_shape = sweep._shape;
		_length = sweep._length;

		sweep._order = 0;
		sweep._shape = nullptr;
		sweep._length = 0;

		return *this;
	}

	constexpr uint64_t DynTensorSweep::getOrder() const
	{
		return _order;
	}

	constexpr const uint64_t* DynTensorSweep::getShape() const
	{
		return _shape;
	}

	constexpr uint64_t DynTensorSweep::getSize(uint64_t i) const
	{
		return _shape[i];
	}

	constexpr uint64_t DynTensorSweep::getTotalLength() const
	{
		return _length;
	}

	constexpr DynTensorSweep::~DynTensorSweep()
	{
		if (_shape)
		{
			delete[] _shape;
		}
	}


	constexpr DynTensorIterator begin(const DynTensorSweep& sweep)
	{
		return DynTensorIterator(&sweep, false);
	}

	constexpr DynTensorIterator end(const DynTensorSweep& sweep)
	{
		return DynTensorIterator(&sweep, true);
	}


	template<uint64_t Order>
	constexpr TensorSweep<Order>::TensorSweep(const uint64_t* shape) :
		_length(std::accumulate(shape, shape + Order, 1, std::multiplies<uint64_t>()))
	{
		assert(_length != 0);
		std::copy(shape, shape + Order, _shape);
	}

	template<uint64_t Order>
	constexpr TensorSweep<Order>::TensorSweep(const std::vector<uint64_t>& shape) : TensorSweep<Order>(shape.data())
	{
		assert(shape.size() == Order);
	}

	template<uint64_t Order>
	constexpr TensorSweep<Order>::TensorSweep(const std::initializer_list<uint64_t>& shape) : TensorSweep<Order>(shape.begin())
	{
		assert(shape.size() == Order);
	}

	template<uint64_t Order>
	constexpr TensorSweep<Order>::TensorSweep(const TensorShaped& tensorShaped) : TensorSweep<Order>(tensorShaped.getShape())
	{
		assert(tensorShaped.getOrder() == Order);
	}

	template<uint64_t Order>
	constexpr TensorSweep<Order>::TensorSweep(const TensorSweep<Order>& sweep) :
		_length(sweep._length)
	{
		std::copy(sweep._shape, sweep._shape + Order, _shape);
	}

	template<uint64_t Order>
	constexpr TensorSweep<Order>& TensorSweep<Order>::operator=(const TensorSweep<Order>& sweep)
	{
		_length = sweep._length;
		std::copy(sweep._shape, sweep._shape + Order, _shape);
	}

	template<uint64_t Order>
	constexpr uint64_t TensorSweep<Order>::getOrder() const
	{
		return Order;
	}

	template<uint64_t Order>
	constexpr const uint64_t* TensorSweep<Order>::getShape() const
	{
		return _shape;
	}

	template<uint64_t Order>
	constexpr uint64_t TensorSweep<Order>::getSize(uint64_t i) const
	{
		return _shape[i];
	}

	template<uint64_t Order>
	constexpr uint64_t TensorSweep<Order>::getTotalLength() const
	{
		return _length;
	}


	template<uint64_t Order>
	constexpr TensorIterator<Order> begin(const TensorSweep<Order>& sweep)
	{
		return TensorIterator<Order>(&sweep, false);
	}

	template<uint64_t Order>
	constexpr TensorIterator<Order> end(const TensorSweep<Order>& sweep)
	{
		return TensorIterator<Order>(&sweep, true);
	}


	template<uint64_t... Shape>
	constexpr uint64_t StaticTensorSweep<Shape...>::getOrder() const
	{
		return _shape.size();
	}

	template<uint64_t... Shape>
	constexpr const uint64_t* StaticTensorSweep<Shape...>::getShape() const
	{
		return _shape.data();
	}

	template<uint64_t... Shape>
	constexpr uint64_t StaticTensorSweep<Shape...>::getSize(uint64_t i) const
	{
		return _shape[i];
	}

	template<uint64_t... Shape>
	constexpr uint64_t StaticTensorSweep<Shape...>::getTotalLength() const
	{
		return (Shape * ...);
	}


	template<uint64_t... Shape>
	constexpr TensorIterator<sizeof...(Shape)> begin(const StaticTensorSweep<Shape...>& sweep)
	{
		return TensorIterator<sizeof...(Shape)>(&sweep, false);
	}

	template<uint64_t... Shape>
	constexpr TensorIterator<sizeof...(Shape)> end(const StaticTensorSweep<Shape...>& sweep)
	{
		return TensorIterator<sizeof...(Shape)>(&sweep, true);
	}
}
