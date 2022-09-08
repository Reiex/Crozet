#pragma once

#include <Diskon/Diskon.hpp>
#include <SciPP/SciPP.hpp>

#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

namespace crz
{
	struct AudioDevice;

	class AudioOutput;


	class SoundSource;

	class SoundBase;
	class SoundFile;
	// TODO: class SoundBuffer;

	class FilterBase;
	class FilterPlaySpeed;
	// TODO: class FilterEnvelope;
}
