#pragma once

#include <Diskon/Core/Checksum.hpp>

namespace dsk
{
	namespace cksm
	{
		namespace _dsk
		{
			template<typename TValue, TValue RevPoly>
			constexpr Crc<TValue, RevPoly>::Crc()
			{
				for (uint16_t i = 0; i < 256; ++i)
				{
					_table[i] = i;
					for (uint8_t j = 0; j < 8; ++j)
					{
						if (_table[i] & 1)
						{
							_table[i] = 0xEDB88320 ^ (_table[i] >> 1);
						}
						else
						{
							_table[i] >>= 1;
						}
					}
				}
			}

			template<typename TValue, TValue RevPoly>
			constexpr TValue Crc<TValue, RevPoly>::operator()(const void* src, uint64_t size, TValue crc) const
			{
				crc = ~crc;

				const uint8_t* it = reinterpret_cast<const uint8_t*>(src);
				const uint8_t* const itEnd = it + size;
				for (; it != itEnd; ++it)
				{
					crc = _table[(crc ^ *it) & 0xFF] ^ (crc >> 8);
				}

				return ~crc;
			}


			template<typename TValue, typename THalf, TValue Modulus>
			constexpr TValue Fletcher<TValue, THalf, Modulus>::operator()(const void* src, uint64_t size, TValue initialValue) const
			{
				constexpr uint8_t halfShift = sizeof(TValue) * 4;
				constexpr TValue lowFilter = std::numeric_limits<TValue>::max() >> halfShift;
				constexpr uint64_t sizeFilter = ~(sizeof(THalf) - 1);

				TValue a = initialValue & lowFilter;
				TValue b = initialValue >> halfShift;

				const THalf* it = reinterpret_cast<const THalf*>(src);
				const THalf* const itEnd = it + (size & sizeFilter);
				for (; it != itEnd; ++it)
				{
					a = (a + *it) % Modulus;
					b = (a + b) % Modulus;
				}

				return (b << halfShift) | a;
			}
		}
	}
}
