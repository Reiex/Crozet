#include <Crozet/Core/Core.hpp>

namespace crz
{
	SoundFile::SoundFile(const std::filesystem::path& path, SoundFileFormat format) :
		_format(format),
		_stream(nullptr)
	{
		switch (_format)
		{
			case SoundFileFormat::Wave:
			{
				dsk::fmt::WaveIStream* waveIStream = new dsk::fmt::WaveIStream();
				waveIStream->setSource(path);

				dsk::fmt::wave::Header header;
				waveIStream->readHeader(header);

				_frequency = header.metadata.frequency;
				_channelCount = header.metadata.channelCount;
				_sampleCount = header.blockCount;

				_stream = waveIStream;

				break;
			}
		}
	}

	void SoundFile::getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		if (!_stream->getError())
		{
			std::fill_n(samples, (timeTo - timeFrom) * _channelCount, 0);
			return;
		}

		switch (_format)
		{
			case SoundFileFormat::Wave:
			{
				dsk::fmt::WaveIStream* waveIStream = dynamic_cast<dsk::fmt::WaveIStream*>(_stream);
				waveIStream->readSampleBlocks(samples, timeTo - timeFrom);
				break;
			}
		}
	}

	SoundFile::~SoundFile()
	{
		delete _stream;
	}
}
