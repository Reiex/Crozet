///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreTypes.hpp>
#include <Crozet/Core/FilterBase.hpp>

namespace crz
{
	class CRZ_API FilterPlaySpeed : public FilterBase
	{
		public:

			FilterPlaySpeed(double speedRatio);
			FilterPlaySpeed(const FilterPlaySpeed& filter) = delete;
			FilterPlaySpeed(FilterPlaySpeed&& filter) = delete;

			FilterPlaySpeed& operator=(const FilterPlaySpeed& filter) = delete;
			FilterPlaySpeed& operator=(FilterPlaySpeed&& filter) = delete;

			virtual uint32_t getFrequency() const override final;
			virtual uint16_t getChannelCount() const override final;
			virtual uint64_t getSampleCount() const override final;
			virtual uint64_t getCurrentSample() const override final;

			virtual ~FilterPlaySpeed() = default;

		private:

			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override final;

			double _speedRatio;
	};
}
