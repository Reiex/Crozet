#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	FilterBase::FilterBase() :
		_source(nullptr)
	{
	}

	void FilterBase::setSource(SoundSource* source)
	{
		_source = source;
	}
}
