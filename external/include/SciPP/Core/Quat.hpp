#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/types.hpp>
#include <SciPP/Core/Tensor/StaticTensor.hpp>

namespace scp
{
	template<typename TValue>
	class Quat
	{
		public:

			constexpr Quat() = default;
			constexpr Quat(const TValue& value);
			constexpr Quat(const TValue& wValue, const TValue& xValue, const TValue& yValue, const TValue& zValue);
			constexpr Quat(const Quat<TValue>& q) = default;
			constexpr Quat(Quat<TValue>&& q) = default;

			constexpr Quat<TValue>& operator=(const Quat<TValue>& q) = default;
			constexpr Quat<TValue>& operator=(Quat<TValue>&& q) = default;

			constexpr Quat<TValue>& operator+=(const Quat<TValue>& q);
			constexpr Quat<TValue>& operator-=(const Quat<TValue>& q);
			constexpr Quat<TValue>& operator*=(const Quat<TValue>& q);
			constexpr Quat<TValue>& operator+=(const TValue& value);
			constexpr Quat<TValue>& operator-=(const TValue& value);
			constexpr Quat<TValue>& operator*=(const TValue& value);
			constexpr Quat<TValue>& operator/=(const TValue& value);

			constexpr bool operator==(const Quat<TValue>& q) const;
			constexpr bool operator!=(const Quat<TValue>& q) const;

			constexpr void setFromRotationAxisAngle(const std::initializer_list<TValue>& axis, const TValue& angle);
			constexpr void setFromRotationAxisAngle(const TensorBase<TValue>& axis, const TValue& angle);
			constexpr void getRotationAxisAngle(TensorBase<TValue>& axis, TValue& angle);

			constexpr void setFromRotationMatrix(const std::initializer_list<TValue>& matrix);
			constexpr void setFromRotationMatrix(const TensorBase<TValue>& matrix);
			constexpr void getRotationMatrix(TensorBase<TValue>& matrix) const;

			constexpr Quat<TValue>& inverse();
			constexpr Quat<TValue>& conjugate();
			constexpr TValue normSq() const;
			constexpr TValue norm() const;

			constexpr ~Quat() = default;

			TValue w;
			TValue x;
			TValue y;
			TValue z;

		private:

			static const TValue _zero;
			static const TValue _one;
	};

	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& p, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& p, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(const Quat<TValue>& p, Quat<TValue>&& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& p, Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue> operator+(const TValue& value, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(const TValue& value, Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& p, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& p, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(const Quat<TValue>& p, Quat<TValue>&& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& p, Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue> operator-(const TValue& value, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(const TValue& value, Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator*(const Quat<TValue>& p, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue> operator*(Quat<TValue>&& p, const Quat<TValue>& q);

	template<typename TValue>
	constexpr Quat<TValue> operator*(const Quat<TValue>& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue>&& operator*(Quat<TValue>&& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue> operator*(const TValue& value, const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator*(const TValue& value, Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator/(const Quat<TValue>& q, const TValue& value);
	template<typename TValue>
	constexpr Quat<TValue>&& operator/(Quat<TValue>&& q, const TValue& value);

	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& q);

	template<typename TValue>
	constexpr Quat<TValue> inverse(const Quat<TValue>& q);
	template<typename TValue>
	constexpr Quat<TValue> conjugate(const Quat<TValue>& q);
}

#include <SciPP/Core/templates/Quat.hpp>