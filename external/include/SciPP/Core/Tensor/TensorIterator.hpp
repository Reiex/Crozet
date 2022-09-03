#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/TensorShaped.hpp>

namespace scp
{
	class DynTensorIterator : public TensorIteratorBase
	{
		public:

			constexpr DynTensorIterator(const TensorShaped* tensorShaped, bool end);
			constexpr DynTensorIterator(const DynTensorIterator& iterator);
			constexpr DynTensorIterator(DynTensorIterator&& iterator);

			constexpr DynTensorIterator& operator=(const DynTensorIterator& iterator);
			constexpr DynTensorIterator& operator=(DynTensorIterator&& iterator);

			constexpr virtual ~DynTensorIterator();
	};

	template<uint64_t Order>
	class TensorIterator : public TensorIteratorBase
	{
		public:

			constexpr TensorIterator(const TensorShaped* tensorShaped, bool end);
			constexpr TensorIterator(const TensorIterator<Order>& iterator);
			constexpr TensorIterator(TensorIterator<Order>&& iterator);

			constexpr TensorIterator<Order>& operator=(const TensorIterator<Order>& iterator);
			constexpr TensorIterator<Order>& operator=(TensorIterator<Order>&& iterator);

			constexpr virtual ~TensorIterator() = default;

		private:

			uint64_t _indices[Order];
	};

	constexpr DynTensorIterator begin(const TensorShaped& tensorShaped);
	constexpr DynTensorIterator end(const TensorShaped& tensorShaped);
}

#include <SciPP/Core/Tensor/templates/TensorIterator.hpp>
