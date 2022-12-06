///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/Core/Mat.hpp>

namespace scp
{
	namespace _scp
	{
		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow>::Mat(TValue diagValue)
		{
			constexpr uint8_t size = Row * Col;
			std::memset(_rows, 0, size * sizeof(TValue));

			constexpr uint8_t n = std::min(Row, Col);

			for (uint8_t i = 0; i < n; ++i)
			{
				_rows[i][i] = diagValue;
			}
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow>::Mat(const std::initializer_list<TValue>& tab)
		{
			constexpr uint8_t size = Row * Col;
			assert(size == tab.size());

			std::memcpy(_rows, tab.begin(), size * sizeof(TValue));
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr TRow& Mat<TValue, Row, Col, TRow>::operator[](uint8_t i)
		{
			assert(i < Row);
			return _rows[i];
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr const TRow& Mat<TValue, Row, Col, TRow>::operator[](uint8_t i) const
		{
			assert(i < Row);
			return _rows[i];
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow>& Mat<TValue, Row, Col, TRow>::operator+=(const Mat<TValue, Row, Col, TRow>& mat)
		{
			for (uint8_t i = 0; i < Row; ++i)
			{
				_rows[i] += mat._rows[i];
			}

			return *this;
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow>& Mat<TValue, Row, Col, TRow>::operator-=(const Mat<TValue, Row, Col, TRow>& mat)
		{
			for (uint8_t i = 0; i < Row; ++i)
			{
				_rows[i] -= mat._rows[i];
			}

			return *this;
		}


		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow> operator+(const Mat<TValue, Row, Col, TRow>& a, const Mat<TValue, Row, Col, TRow>& b)
		{
			Mat<TValue, Row, Col, TRow> c(a);
			return c += b;
		}

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow> operator-(const Mat<TValue, Row, Col, TRow>& a, const Mat<TValue, Row, Col, TRow>& b)
		{
			Mat<TValue, Row, Col, TRow> c(a);
			return c -= b;
		}

		template<typename TValue, uint8_t RowA, uint8_t ColARowB, uint8_t ColB, typename TRowA, typename TRowB>
		constexpr Mat<TValue, RowA, ColB, TRowB> operator*(const Mat<TValue, RowA, ColB, TRowB>& a, const Mat<TValue, RowA, ColB, TRowB>& b)
		{
			Mat<TValue, RowA, ColB, TRowB> c(0);

			for (uint8_t i = 0; i < RowA; ++i)
			{
				for (uint8_t j = 0; j < ColB; ++j)
				{
					for (uint8_t k = 0; k < ColARowB; ++k)
					{
						c[i][j] += a[i][k] * b[k][j];
					}
				}
			}

			return c;
		}
	}
}
