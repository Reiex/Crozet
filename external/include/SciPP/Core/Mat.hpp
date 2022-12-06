///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/Core/types.hpp>

namespace scp
{
	namespace _scp
	{
		#pragma pack(push, 1)
		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		class Mat
		{
			public:

				constexpr Mat() = default;
				constexpr Mat(TValue diagValue);
				constexpr Mat(const std::initializer_list<TValue>& values);
				constexpr Mat(const Mat<TValue, Row, Col, TRow>& mat) = default;
				constexpr Mat(Mat<TValue, Row, Col, TRow>&& mat) = default;

				constexpr Mat<TValue, Row, Col, TRow>& operator=(const Mat<TValue, Row, Col, TRow>& mat) = default;
				constexpr Mat<TValue, Row, Col, TRow>& operator=(Mat<TValue, Row, Col, TRow>&& mat) = default;

				constexpr TRow& operator[](uint8_t i);
				constexpr const TRow& operator[](uint8_t i) const;

				constexpr Mat<TValue, Row, Col, TRow>& operator+=(const Mat<TValue, Row, Col, TRow>& mat);
				constexpr Mat<TValue, Row, Col, TRow>& operator-=(const Mat<TValue, Row, Col, TRow>& mat);

				constexpr ~Mat() = default;

			private:

				TRow _rows[Row];
		};
		#pragma pack(pop)

		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow> operator+(const Mat<TValue, Row, Col, TRow>& a, const Mat<TValue, Row, Col, TRow>& b);
		template<typename TValue, uint8_t Row, uint8_t Col, typename TRow>
		constexpr Mat<TValue, Row, Col, TRow> operator-(const Mat<TValue, Row, Col, TRow>& a, const Mat<TValue, Row, Col, TRow>& b);

		template<typename TValue, uint8_t RowA, uint8_t ColARowB, uint8_t ColB, typename TRowA, typename TRowB>
		constexpr Mat<TValue, RowA, ColB, TRowB> operator*(const Mat<TValue, RowA, ColARowB, TRowA>& a, const Mat<TValue, ColARowB, ColB, TRowB>& b);

		// TODO: matrix-vector product, inverse, determinant, unary minus
	}
}

#include <SciPP/Core/templates/Mat.hpp>
