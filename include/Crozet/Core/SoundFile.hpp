#pragma once

#include <Crozet/Core/types.hpp>
#include <Crozet/Core/SoundBase.hpp>

namespace crz
{
	enum class SoundFileFormat
	{
		Wave
	};

	class SoundFile : public SoundBase
	{
		public:

			SoundFile(const std::filesystem::path& path, SoundFileFormat format);
			SoundFile(const SoundFile& sound) = delete;
			SoundFile(SoundFile&& sound) = delete;

			SoundFile& operator=(const SoundFile& sound) = delete;
			SoundFile& operator=(SoundFile&& sound) = delete;

			~SoundFile();

		private:

			void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override final;

			SoundFileFormat _format;
			dsk::fmt::FormatIStream* _stream;
	};
}
