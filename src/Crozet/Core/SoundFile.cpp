///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	namespace
	{
		inline uint64_t crzRead(void* handle, uint8_t* data, uint64_t size)
		{
			return std::fread(data, 1, size, reinterpret_cast<std::FILE*>(handle));
		}

		inline bool crzEof(void* handle)
		{
			return std::feof(reinterpret_cast<std::FILE*>(handle));
		}
	}

	SoundFile::SoundFile(const std::filesystem::path& path, SoundFileFormat format) : SoundBase(),
		_format(format),
		_file(nullptr),
		_stream(nullptr),
		_formatStream(nullptr)
	{
		if (!std::filesystem::exists(path))
		{
			return;
		}

		_file = std::fopen(path.string().c_str(), "rb");
		if (!_file)
		{
			return;
		}

		_stream = new dsk::IStream(_file, crzRead, crzEof);

		switch (_format)
		{
			case SoundFileFormat::Wave:
			{
				dsk::fmt::WaveIStream* waveIStream = new dsk::fmt::WaveIStream(_stream);
				
				dsk::fmt::wave::Header header;
				waveIStream->readHeader(header);

				_frequency = header.metadata.frequency;
				_channelCount = header.metadata.channelCount;
				_sampleCount = header.blockCount;

				_formatStream = waveIStream;

				break;
			}
		}
	}

	void SoundFile::getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		if (!_stream->getStatus())
		{
			std::fill_n(samples, (timeTo - timeFrom) * _channelCount, 0);
			return;
		}

		switch (_format)
		{
			case SoundFileFormat::Wave:
			{
				dsk::fmt::WaveIStream* waveIStream = dynamic_cast<dsk::fmt::WaveIStream*>(_formatStream);

				if (timeFrom != _currentSample)
				{
					waveIStream->skipBlocks(timeFrom - _currentSample);
				}

				waveIStream->readSampleBlocks(samples, timeTo - timeFrom);
				break;
			}
		}

		_currentSample = timeTo;
	}

	SoundFile::~SoundFile()
	{
		delete _formatStream;
		delete _stream;
		std::fclose(_file);
	}
}
