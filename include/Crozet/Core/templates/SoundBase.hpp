///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author P�l�grin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreDecl.hpp>

namespace crz
{
	template<std::derived_from<FilterBase> TFilter, typename... Args>
	uint64_t SoundBase::addFilter(Args&&... args)
	{
		assert(_currentSample == 0);

		const uint64_t index = _filters.size();

		FilterBase* filter = new TFilter(std::forward<Args>(args)...);
		if (index == 0)
		{
			filter->setSource(this);
		}
		else
		{
			filter->setSource(_filters.back());
		}

		_filters.push_back(filter);

		return index;
	}
}
