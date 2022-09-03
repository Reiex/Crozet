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
#include <SciPP/Core/Tensor/TensorSweep.hpp>

namespace scp
{
	enum class ConvolutionMethod
	{
		Zero,
		Continuous,
		Periodic
	};

	enum class InterpolationMethod
	{
		Nearest,
		Linear,
		Cubic
	};


	template<typename TValue>
	class TensorBase : public TensorShaped
	{
		public:

			using ValueType = TValue;

			TensorBase(const TensorBase<TValue>& tensor) = delete;
			TensorBase(TensorBase<TValue>&& tensor) = delete;

			TensorBase<TValue>& operator=(const TensorBase<TValue>& tensor) = delete;
			TensorBase<TValue>& operator=(TensorBase<TValue>&& tensor) = delete;

			constexpr virtual TensorBase<TValue>* clone() const = 0;

			constexpr virtual TensorBase<TValue>& operator+=(const TensorBase<TValue>& tensor);
			constexpr virtual TensorBase<TValue>& operator-=(const TensorBase<TValue>& tensor);
			constexpr virtual TensorBase<TValue>& operator*=(const TValue& value);
			constexpr virtual TensorBase<TValue>& operator/=(const TValue& value);

			constexpr virtual bool operator==(const TensorBase<TValue>& tensor) const;
			constexpr bool operator!=(const TensorBase<TValue>& tensor) const;

			constexpr virtual TensorBase<TValue>& hadamardProduct(const TensorBase<TValue>& tensor);
			constexpr virtual TensorBase<TValue>& convolution(const TensorBase<TValue>& kernel, ConvolutionMethod method);
			constexpr virtual TensorBase<TValue>& fft();
			constexpr virtual TensorBase<TValue>& ifft();
			constexpr virtual TensorBase<TValue>& negate();
			constexpr virtual TValue dotProduct(const TensorBase<TValue>& tensor) const;
			constexpr virtual void tensorContraction(uint64_t i, uint64_t j, TensorBase<TValue>& result) const;
			constexpr virtual void tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const;
			constexpr virtual void contractedTensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const;
			constexpr virtual void interpolation(TensorBase<TValue>& result, InterpolationMethod method) const;

			constexpr virtual TensorBase<TValue>& fill(const TValue& value);
			constexpr virtual TensorBase<TValue>& apply(const std::function<TValue(const TValue&)>& function);

			constexpr virtual TValue normSq() const;
			constexpr virtual TValue norm() const final;
			constexpr virtual const TValue& minElement() const;
			constexpr virtual const TValue& maxElement() const;

			constexpr virtual const TValue& get(uint64_t internalIndex) const = 0;
			constexpr virtual const TValue& get(const uint64_t* indices) const = 0;
			constexpr virtual const TValue& get(const std::initializer_list<uint64_t>& indices) const = 0;
			constexpr virtual void set(uint64_t internalIndex, const TValue& value) = 0;
			constexpr virtual void set(const uint64_t* indices, const TValue& value) = 0;
			constexpr virtual void set(const std::initializer_list<uint64_t>& indices, const TValue& value) = 0;

			constexpr virtual void getIndices(uint64_t internalIndex, uint64_t* indices) const override = 0;
			constexpr virtual uint64_t getInternalIndex(const uint64_t* indices) const override = 0;
			constexpr virtual uint64_t getInternalLength() const override = 0;

			constexpr virtual uint64_t getOrder() const override = 0;
			constexpr virtual const uint64_t* getShape() const override = 0;
			constexpr virtual uint64_t getSize(uint64_t i) const override = 0;

			constexpr virtual ~TensorBase() = default;
		
		protected:

			static const TValue _zero;

			constexpr TensorBase() = default;

			constexpr virtual void setInitialPosition(TensorIteratorBase * iterator, bool end) const override = 0;
			constexpr virtual void incrementPosition(TensorIteratorBase * iterator) const override = 0;

			constexpr virtual void create(uint64_t order, const uint64_t* shape) = 0;
			constexpr virtual void copyFrom(const TensorBase<TValue>& tensor);
			constexpr virtual void moveFrom(TensorBase<TValue>&& tensor);
			constexpr virtual void destroy() = 0;
	};

	template<TensorConcept TTensor>
	constexpr TTensor operator+(const TTensor& a, const TTensor& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& a, const TTensor& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(const TTensor& a, TTensor&& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& a, TTensor&& b);

	template<TensorConcept TTensor>
	constexpr TTensor operator-(const TTensor& a, const TTensor& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& a, const TTensor& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(const TTensor& a, TTensor&& b);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& a, TTensor&& b);

	template<TensorConcept TTensor>
	constexpr TTensor operator*(const TTensor& tensor, const typename TTensor::ValueType& value);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator*(TTensor&& tensor, const typename TTensor::ValueType& value);
	template<TensorConcept TTensor>
	constexpr TTensor operator*(const typename TTensor::ValueType& value, const TTensor& tensor);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator*(const typename TTensor::ValueType& value, TTensor&& tensor);

	template<TensorConcept TTensor>
	constexpr TTensor operator/(const TTensor& tensor, const typename TTensor::ValueType& value);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator/(TTensor&& tensor, const typename TTensor::ValueType& value);

	template<TensorConcept TTensor>
	constexpr TTensor operator-(const TTensor& a);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& a);

	template<TensorConcept TTensor>
	constexpr TTensor operator+(const TTensor& a);
	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& a);


	template<TensorConcept TTensor>
	constexpr TTensor hadamardProduct(const TTensor& a, const TTensor& b);
	template<TensorConcept TTensor>
	constexpr TTensor convolution(const TTensor& tensor, const TensorBase<typename TTensor::ValueType>& kernel, ConvolutionMethod method);
	template<TensorConcept TTensor>
	constexpr TTensor fft(const TTensor& tensor);
	template<TensorConcept TTensor>
	constexpr TTensor ifft(const TTensor& tensor);
	template<typename TValue>
	constexpr typename TValue dotProduct(const TensorBase<TValue>& a, const TensorBase<TValue>& b);
}

#include <SciPP/Core/Tensor/templates/TensorBase.hpp>
