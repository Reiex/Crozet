///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreTypes.hpp>
#include <Crozet/Core/SoundBase.hpp>

namespace crz
{
	enum class SoundFileFormat
	{
		Wave
	};

	class CRZ_API SoundFile : public SoundBase
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

			std::FILE* _file;
			dsk::IStream* _stream;
			dsk::fmt::FormatIStream* _formatStream;
	};
}
