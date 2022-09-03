#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/TensorShaped.hpp>
#include <SciPP/Core/Tensor/TensorIterator.hpp>

namespace scp
{
	class TensorSweepBase : public TensorShaped
	{
		public:

			TensorSweepBase(const TensorSweepBase& sweep) = delete;
			TensorSweepBase(TensorSweepBase&& sweep) = delete;

			TensorSweepBase& operator=(const TensorSweepBase& sweep) = delete;
			TensorSweepBase& operator=(TensorSweepBase&& sweep) = delete;

			constexpr virtual void getIndices(uint64_t internalIndex, uint64_t* indices) const override final;
			constexpr virtual uint64_t getInternalIndex(const uint64_t* indices) const override final;
			constexpr virtual uint64_t getInternalLength() const override final;

		protected:

			constexpr virtual void setInitialPosition(TensorIteratorBase* iterator, bool end) const override final;
			constexpr virtual void incrementPosition(TensorIteratorBase* iterator) const override final;

			TensorSweepBase() = default;
	};


	class DynTensorSweep : public TensorSweepBase
	{
		public:

			DynTensorSweep() = delete;
			constexpr DynTensorSweep(uint64_t order, const uint64_t* shape);
			constexpr DynTensorSweep(const std::vector<uint64_t>& shape);
			constexpr DynTensorSweep(const std::initializer_list<uint64_t>& shape);
			constexpr DynTensorSweep(const TensorShaped& tensorShaped);
			constexpr DynTensorSweep(const DynTensorSweep& sweep);
			constexpr DynTensorSweep(DynTensorSweep&& sweep);

			constexpr DynTensorSweep& operator=(const DynTensorSweep& sweep);
			constexpr DynTensorSweep& operator=(DynTensorSweep&& sweep);

			constexpr virtual uint64_t getOrder() const override final;
			constexpr virtual const uint64_t* getShape() const override final;
			constexpr virtual uint64_t getSize(uint64_t i) const override final;
			constexpr virtual uint64_t getTotalLength() const override final;

			constexpr virtual ~DynTensorSweep();

		private:

			uint64_t _order;
			uint64_t* _shape;
			uint64_t _length;
	};

	constexpr DynTensorIterator begin(const DynTensorSweep& sweep);
	constexpr DynTensorIterator end(const DynTensorSweep& sweep);


	template<uint64_t Order>
	class TensorSweep : public TensorSweepBase
	{
		public:

			TensorSweep() = delete;
			constexpr TensorSweep(const uint64_t* shape);
			constexpr TensorSweep(const std::vector<uint64_t>& shape);
			constexpr TensorSweep(const std::initializer_list<uint64_t>& shape);
			constexpr TensorSweep(const TensorShaped& tensorShaped);
			constexpr TensorSweep(const TensorSweep<Order>& sweep);

			constexpr TensorSweep<Order>& operator=(const TensorSweep<Order>& sweep);

			constexpr virtual uint64_t getOrder() const override final;
			constexpr virtual const uint64_t* getShape() const override final;
			constexpr virtual uint64_t getSize(uint64_t i) const override final;
			constexpr virtual uint64_t getTotalLength() const override final;

			constexpr virtual ~TensorSweep() = default;

		private:

			uint64_t _shape[Order];
			uint64_t _length;
	};

	template<uint64_t Order>
	constexpr TensorIterator<Order> begin(const TensorSweep<Order>& sweep);
	template<uint64_t Order>
	constexpr TensorIterator<Order> end(const TensorSweep<Order>& sweep);


	template<uint64_t... Shape>
	class StaticTensorSweep : public TensorSweepBase
	{
		public:

			constexpr StaticTensorSweep() = default;
			constexpr StaticTensorSweep(const StaticTensorSweep<Shape...>& sweep) = delete;
			constexpr StaticTensorSweep(StaticTensorSweep<Shape...>&& sweep) = delete;

			constexpr StaticTensorSweep<Shape...>& operator=(const StaticTensorSweep<Shape...>& sweep) = delete;
			constexpr StaticTensorSweep<Shape...>& operator=(StaticTensorSweep<Shape...>&& sweep) = delete;

			constexpr virtual uint64_t getOrder() const override final;
			constexpr virtual const uint64_t* getShape() const override final;
			constexpr virtual uint64_t getSize(uint64_t i) const override final;
			constexpr virtual uint64_t getTotalLength() const override final;

			constexpr virtual ~StaticTensorSweep() = default;

		private:

			static constexpr std::array<uint64_t, sizeof...(Shape)> _shape = { Shape... };
	};

	template<uint64_t... Shape>
	constexpr TensorIterator<sizeof...(Shape)> begin(const StaticTensorSweep<Shape...>& sweep);
	template<uint64_t... Shape>
	constexpr TensorIterator<sizeof...(Shape)> end(const StaticTensorSweep<Shape...>& sweep);
}

#include <SciPP/Core/Tensor/templates/TensorSweep.hpp>
