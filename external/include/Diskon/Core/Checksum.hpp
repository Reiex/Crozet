///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Diskon/Core/types.hpp>

namespace dsk
{
	namespace cksm
	{
		// CRC

		namespace _dsk
		{
			template<typename TValue, TValue RevPoly>
			class Crc
			{
				public:

					constexpr Crc();
					constexpr TValue operator()(const void* src, uint64_t size, TValue crc = 0) const;

				private:

					TValue _table[256];
			};
		}

		// Fletcher/Adler

		namespace _dsk
		{
			template<typename TValue, typename THalf, TValue Modulus>
			struct Fletcher
			{
				constexpr Fletcher() = default;
				constexpr TValue operator()(const void* src, uint64_t size, TValue initialValue = 1) const;
			};
		}
	}
}

#include <Diskon/Core/templates/Checksum.hpp>

namespace dsk
{
	namespace cksm
	{
		constexpr _dsk::Crc<uint32_t, 0xEDB88320> crc32;

		constexpr _dsk::Fletcher<uint16_t, uint8_t, 0xFF> fletcher16;
		constexpr _dsk::Fletcher<uint32_t, uint16_t, 0xFFFF> fletcher32;
		constexpr _dsk::Fletcher<uint64_t, uint32_t, 0xFFFFFFFF> fletcher64;
		constexpr _dsk::Fletcher<uint32_t, uint8_t, 65521> adler32;
	}
}
