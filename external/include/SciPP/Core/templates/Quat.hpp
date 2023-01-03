///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/Core/CoreDecl.hpp>

namespace scp
{
	template<typename TValue>
	const TValue Quat<TValue>::_zero(0);

	template<typename TValue>
	const TValue Quat<TValue>::_one(1);

	template<typename TValue>
	constexpr Quat<TValue>::Quat(const TValue& value) :
		w(value),
		x(_zero),
		y(_zero),
		z(_zero)
	{
	}

	template<typename TValue>
	constexpr Quat<TValue>::Quat(const TValue& wValue, const TValue& xValue, const TValue& yValue, const TValue& zValue) :
		w(wValue),
		x(xValue),
		y(yValue),
		z(zValue)
	{
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator+=(const Quat<TValue>& q)
	{
		w += q.w;
		x += q.x;
		y += q.y;
		z += q.z;

		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator-=(const Quat<TValue>& q)
	{
		w -= q.w;
		x -= q.x;
		y -= q.y;
		z -= q.z;

		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator*=(const Quat<TValue>& q)
	{
		const TValue a = w * q.w - x * q.x - y * q.y - z * q.z;
		const TValue b = w * q.x + x * q.w + y * q.z - z * q.y;
		const TValue c = w * q.y - x * q.z + y * q.w + z * q.x;
		const TValue d = w * q.z + x * q.y - y * q.x + z * q.w;

		w = a;
		x = b;
		y = c;
		z = d;

		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator+=(const TValue& value)
	{
		w += value;
		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator-=(const TValue& value)
	{
		w -= value;
		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator*=(const TValue& value)
	{
		w *= value;
		x *= value;
		y *= value;
		z *= value;
		return *this;
	}

	template<typename TValue>
	constexpr Quat<TValue>& Quat<TValue>::operator/=(const TValue& value)
	{
		w /= value;
		x /= value;
		y /= value;
		z /= value;
		return *this;
	}

	template<typename TValue>
	constexpr bool Quat<TValue>::operator==(const Quat<TValue>& q) const
	{
		return w = q.w && x = q.x && y = q.y && z = q.z;
	}

	template<typename TValue>
	constexpr bool Quat<TValue>::operator!=(const Quat<TValue>& q) const
	{
		return !(*this == q);
	}

	template<typename TValue>
	constexpr void Quat<TValue>::setFromRotationAxisAngle(const std::initializer_list<TValue>& axis, const TValue& angle)
	{
		assert(axis.size() == 3);

		const TValue& xAxis = axis.begin()[0];
		const TValue& yAxis = axis.begin()[1];
		const TValue& zAxis = axis.begin()[2];

		assert(std::abs(xAxis * xAxis + yAxis * yAxis + zAxis * zAxis - _one) < FLT_EPSILON);

		const TValue halfAngle = angle / 2;
		const TValue sinHalfAngle = std::sin(halfAngle);

		w = std::cos(halfAngle);
		x = xAxis * sinHalfAngle;
		y = yAxis * sinHalfAngle;
		z = zAxis * sinHalfAngle;
	}

	template<typename TValue>
	template<CTensor<TValue> TTensor>
	constexpr void Quat<TValue>::setFromRotationAxisAngle(const TTensor& axis, const TValue& angle)
	{
		assert(axis.getOrder() == 1);
		assert(axis.getSize(0) == 3);
		setFromAxisAngle({ axis.get(0ULL), axis.get(1ULL), axis.get(2ULL) }, angle);
	}

	template<typename TValue>
	template<CTensor<TValue> TTensor>
	constexpr void Quat<TValue>::getRotationAxisAngle(TTensor& axis, TValue& angle)
	{
		assert(axis.getOrder() == 1);
		assert(axis.getSize(0) == 3);
		assert(normSq() - _one < FLT_EPSILON);

		angle = std::acos(w) * 2;

		const TValue factor = std::sin(angle / 2);
		axis.set(0ULL, x / factor);
		axis.set(1ULL, y / factor);
		axis.set(2ULL, z / factor);
	}

	template<typename TValue>
	constexpr void Quat<TValue>::setFromRotationMatrix(const std::initializer_list<TValue>& matrix)
	{
		assert(matrix.size() == 9);
		// TODO: Verify it is a rotation matrix

		const TValue& m00 = matrix[0];
		const TValue& m01 = matrix[1];
		const TValue& m02 = matrix[2];
		const TValue& m10 = matrix[3];
		const TValue& m11 = matrix[4];
		const TValue& m12 = matrix[5];
		const TValue& m20 = matrix[6];
		const TValue& m21 = matrix[7];
		const TValue& m22 = matrix[8];

		const TValue trace = m00 + m11 + m22;

		if (trace > _zero)
		{
			const TValue s = 2 * std::sqrt(_one + trace);
			w = s / 4;
			x = (m21 - m12) / s;
			y = (m02 - m20) / s;
			z = (m10 - m01) / s;
		}
		else if (m00 > m11 && m00 > m22)
		{
			const TValue s = 2 * std::sqrt(_one + m00 - m11 - m22);
			w = (m21 - m12) / s;
			x = s / 4;
			y = (m01 + m10) / s;
			z = (m02 + m20) / s;
		}
		else if (m11 > m22)
		{
			const TValue s = 2 * std::sqrt(_one - m00 + m11 - m22);
			w = (m02 - m20) / s;
			x = (m01 + m10) / s;
			y = s / 4;
			z = (m12 + m21) / s;
		}
		else
		{
			const TValue s = 2 * std::sqrt(_one - m00 - m11 + m22);
			w = (m10 - m01) / s;
			x = (m02 + m20) / s;
			y = (m12 + m21) / s;
			z = s / 4;
		}
	}

	template<typename TValue>
	template<CTensor<TValue> TTensor>
	constexpr void Quat<TValue>::setFromRotationMatrix(const TTensor& matrix)
	{
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == 3);
		assert(matrix.getSize(1) == 3);
		setFromMatrix(
			{
				matrix.get(0ULL), matrix.get(1ULL), matrix.get(2ULL),
				matrix.get(3ULL), matrix.get(4ULL), matrix.get(5ULL),
				matrix.get(6ULL), matrix.get(7ULL), matrix.get(8ULL)
			}
		);
	}

	template<typename TValue>
	template<CTensor<TValue> TTensor>
	constexpr void Quat<TValue>::getRotationMatrix(TTensor& matrix) const
	{
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == 3);
		assert(matrix.getSize(1) == 3);
		assert(normSq() - _one < FLT_EPSILON);

		const TValue ww = w * w;
		const TValue xx = x * x;
		const TValue yy = y * y;
		const TValue zz = z * z;

		const TValue wx2 = 2 * w * x;
		const TValue wy2 = 2 * w * y;
		const TValue wz2 = 2 * w * z;
		const TValue xy2 = 2 * x * y;
		const TValue xz2 = 2 * x * z;
		const TValue yz2 = 2 * y * z;

		matrix.set(0ULL, ww + xx - yy - zz);
		matrix.set(1ULL, xy2 - wz2);
		matrix.set(2ULL, wy2 + xz2);
		matrix.set(3ULL, wz2 + xy2);
		matrix.set(4ULL, ww - xx + yy - zz);
		matrix.set(5ULL, yz2 - wx2);
		matrix.set(6ULL, xz2 - wy2);
		matrix.set(7ULL, wx2 + yz2);
		matrix.set(8ULL, ww - xx - yy + zz);
	}

	template<typename TValue>
	constexpr void Quat<TValue>::inverse()
	{
		assert(w != _zero || x != _zero || y != _zero || z != _zero);

		const TValue length = normSq();
		w /= length;
		x = -x / length;
		y = -y / length;
		z = -z / length;
	}

	template<typename TValue>
	constexpr void Quat<TValue>::conjugate()
	{
		x = -x;
		y = -y;
		z = -z;
	}

	template<typename TValue>
	constexpr TValue Quat<TValue>::normSq() const
	{
		return w * w + x * x + y * y + z * z;
	}

	template<typename TValue>
	constexpr TValue Quat<TValue>::norm() const
	{
		return std::sqrt(normSq());
	}


	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& p, const Quat<TValue>& q)
	{
		Quat<TValue> result(p);
		result += q;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& p, const Quat<TValue>& q)
	{
		p += q;
		return std::move(p);
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(const Quat<TValue>& p, Quat<TValue>&& q)
	{
		q += p;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& p, Quat<TValue>&& q)
	{
		p += q;
		return std::move(p);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& q, const TValue& value)
	{
		Quat<TValue> result(q);
		result += value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& q, const TValue& value)
	{
		q += value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator+(const TValue& value, const Quat<TValue>& q)
	{
		Quat<TValue> result(q);
		result += value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(const TValue& value, Quat<TValue>&& q)
	{
		q += value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& p, const Quat<TValue>& q)
	{
		Quat<TValue> result(p);
		result -= q;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& p, const Quat<TValue>& q)
	{
		p -= q;
		return std::move(p);
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(const Quat<TValue>& p, Quat<TValue>&& q)
	{
		q -= p;
		return -std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& p, Quat<TValue>&& q)
	{
		p -= q;
		return std::move(p);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& q, const TValue& value)
	{
		Quat<TValue> result(q);
		result -= value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& q, const TValue& value)
	{
		q -= value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator-(const TValue& value, const Quat<TValue>& q)
	{
		Quat<TValue> result(q);
		result -= value;
		return -result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(const TValue& value, Quat<TValue>&& q)
	{
		q -= value;
		return -std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator*(const Quat<TValue>& p, const Quat<TValue>& q)
	{
		Quat<TValue> result(p);
		result *= q;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue> operator*(Quat<TValue>&& p, const Quat<TValue>& q)
	{
		p *= q;
		return std::move(p);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator*(const Quat<TValue>& q, const TValue& value)
	{
		Quat<TValue> result(q);
		result *= value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator*(Quat<TValue>&& q, const TValue& value)
	{
		q *= value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator*(const TValue& value, const Quat<TValue>& q)
	{
		Quat<TValue> result(q);
		result *= value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator*(const TValue& value, Quat<TValue>&& q)
	{
		q *= value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator/(const Quat<TValue>& q, const TValue& value)
	{
		Quat<TValue> result(q);
		result /= value;
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator/(Quat<TValue>&& q, const TValue& value)
	{
		q /= value;
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator+(const Quat<TValue>& q)
	{
		return q;
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator+(Quat<TValue>&& q)
	{
		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> operator-(const Quat<TValue>& q)
	{
		return { -q.w, -q.x, -q.y, -q.z };
	}

	template<typename TValue>
	constexpr Quat<TValue>&& operator-(Quat<TValue>&& q)
	{
		q.w = -q.w;
		q.x = -q.x;
		q.y = -q.y;
		q.z = -q.z;

		return std::move(q);
	}

	template<typename TValue>
	constexpr Quat<TValue> inverse(const Quat<TValue>& q)
	{
		Quat<TValue> result(q);
		result.inverse();
		return result;
	}

	template<typename TValue>
	constexpr Quat<TValue> conjugate(const Quat<TValue>& q)
	{
		Quat<TValue> result(q);
		result.conjugate();
		return result;
	}
}
