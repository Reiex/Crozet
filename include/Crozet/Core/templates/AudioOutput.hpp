///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/AudioOutput.hpp>

namespace crz
{
	template<std::derived_from<SoundBase> TSound, typename... Args>
	uint64_t AudioOutput::createSound(Args&&... args)
	{
		assert(isValid());

		_sounds.emplace(_nextSoundId, new TSound(std::forward<Args>(args)...));
		return _nextSoundId++;
	}
}
