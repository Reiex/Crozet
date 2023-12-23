///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélérin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreTypes.hpp>
#include <Crozet/Core/SoundSource.hpp>

namespace crz
{
	class CRZ_API FilterBase : public SoundSource
	{
		public:

			FilterBase(const FilterBase& filter) = delete;
			FilterBase(FilterBase&& filter) = delete;

			FilterBase& operator=(const FilterBase& filter) = delete;
			FilterBase& operator=(FilterBase&& filter) = delete;

			virtual uint32_t getFrequency() const override = 0;
			virtual uint16_t getChannelCount() const override = 0;
			virtual uint64_t getSampleCount() const override = 0;
			virtual uint64_t getCurrentSample() const override = 0;

			void setSource(SoundSource* source);

			virtual ~FilterBase() = default;

		protected:

			FilterBase();

			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override = 0;

			SoundSource* _source;
	};
}
