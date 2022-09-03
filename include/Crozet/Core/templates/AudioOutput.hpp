#pragma once

#include <Crozet/Core/AudioOutput.hpp>

namespace crz
{
	template<std::derived_from<SoundBase> TSound, typename... Args>
	uint64_t AudioOutput::createSound(Args&&... args)
	{
		_sounds.emplace(_nextSoundIndex, new TSound(std::forward<Args>(args)...));
		return _nextSoundIndex++;
	}
}
