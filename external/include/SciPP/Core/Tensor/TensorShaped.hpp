#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>

namespace scp
{
	struct TensorPosition
	{
		uint64_t* indices;
		uint64_t internalIndex;
	};


	class TensorIteratorBase
	{
		public:

			constexpr TensorIteratorBase(const TensorShaped* tensorShaped, uint64_t* indicesPtr, bool end);
			constexpr TensorIteratorBase(const TensorIteratorBase& iterator, uint64_t* indicesPtr);
			constexpr TensorIteratorBase(TensorIteratorBase&& iterator, uint64_t* indicesPtr);
			TensorIteratorBase(const TensorIteratorBase& iterator) = delete;
			TensorIteratorBase(TensorIteratorBase&& iterator) = delete;

			TensorIteratorBase& operator=(const TensorIteratorBase& iterator) = delete;
			TensorIteratorBase& operator=(TensorIteratorBase&& iterator) = delete;

			constexpr const TensorPosition& operator*() const;
			constexpr const TensorPosition* operator->() const;

			constexpr TensorIteratorBase& operator++();

			constexpr bool operator==(const TensorIteratorBase& iterator) const;
			constexpr bool operator!=(const TensorIteratorBase& iterator) const;

			constexpr const uint64_t* getIndices() const;
			constexpr uint64_t* getIndices();
			constexpr const uint64_t& getInternalIndex() const;
			constexpr uint64_t& getInternalIndex();
			constexpr const uint64_t& getSavedIndex() const;
			constexpr uint64_t& getSavedIndex();

			constexpr virtual ~TensorIteratorBase() = default;

		protected:

			constexpr void copyFrom(const TensorIteratorBase& iterator, uint64_t* indicesPtr);
			constexpr void moveFrom(TensorIteratorBase&& iterator, uint64_t* indicesPtr);

			const TensorShaped* _tensorShaped;
			TensorPosition _position;

			uint64_t _order;
			const uint64_t* _shape;
			uint64_t _internalCounter;
			uint64_t _savedIndex;

	};


	class TensorShaped
	{
		public:

			TensorShaped(const TensorShaped& tensor) = delete;
			TensorShaped(TensorShaped&& tensor) = delete;

			TensorShaped& operator=(const TensorShaped& tensor) = delete;
			TensorShaped& operator=(TensorShaped&& tensor) = delete;

			constexpr virtual void getIndices(uint64_t internalIndex, uint64_t* indices) const = 0;
			constexpr virtual uint64_t getInternalIndex(const uint64_t* indices) const = 0;
			constexpr virtual uint64_t getInternalLength() const = 0;

			constexpr virtual uint64_t getOrder() const = 0;
			constexpr virtual const uint64_t* getShape() const = 0;
			constexpr virtual uint64_t getSize(uint64_t i) const = 0;
			constexpr virtual uint64_t getTotalLength() const;

			constexpr virtual ~TensorShaped() = default;

		protected:

			constexpr virtual void setInitialPosition(TensorIteratorBase* iterator, bool end) const = 0;
			constexpr virtual void incrementPosition(TensorIteratorBase* iterator) const = 0;

			constexpr TensorShaped() = default;

		friend class TensorIteratorBase;
	};
}

#include <SciPP/Core/Tensor/templates/TensorShaped.hpp>
