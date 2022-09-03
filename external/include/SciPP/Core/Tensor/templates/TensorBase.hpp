#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/TensorBase.hpp>

namespace scp
{
	template<typename TValue>
	const TValue TensorBase<TValue>::_zero(0);

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::operator+=(const TensorBase<TValue>& tensor)
	{
		assert(getOrder() == tensor.getOrder());
		assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, get(pos.indices) + tensor.get(pos.indices));
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::operator-=(const TensorBase<TValue>& tensor)
	{
		assert(getOrder() == tensor.getOrder());
		assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, get(pos.indices) - tensor.get(pos.indices));
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::operator*=(const TValue& value)
	{
		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, get(pos.indices) * value);
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::operator/=(const TValue& value)
	{
		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, get(pos.indices) / value);
		}

		return *this;
	}

	template<typename TValue>
	constexpr bool TensorBase<TValue>::operator==(const TensorBase<TValue>& tensor) const
	{
		const uint64_t order = getOrder();
		if (order != tensor.getOrder())
		{
			return false;
		}

		const uint64_t* shape = getShape();
		if (!std::equal(shape, shape + order, tensor.getShape()))
		{
			return false;
		}

		for (const TensorPosition& pos : *this)
		{
			if (get(pos.indices) != tensor.get(pos.indices))
			{
				return false;
			}
		}

		return true;
	}

	template<typename TValue>
	constexpr bool TensorBase<TValue>::operator!=(const TensorBase<TValue>& tensor) const
	{
		return !(*this == tensor);
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::hadamardProduct(const TensorBase<TValue>& tensor)
	{
		assert(getOrder() == tensor.getOrder());
		assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, get(pos.indices) * tensor.get(pos.indices));
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::convolution(const TensorBase<TValue>& kernel, ConvolutionMethod method)
	{
		assert(getOrder() == kernel.getOrder());
		
		// Do a copy of *this
		TensorBase<TValue>* tensor = clone();
		const uint64_t order = tensor->getOrder();
		const uint64_t* shape = tensor->getShape();

		// Check that the kernel's shape is odd
		for (uint64_t i = 0; i < order; i++)
		{
			assert(kernel.getSize(i) % 2 == 1);
			assert(kernel.getSize(i) <= getSize(i));
		}

		// Compute offset (to center the kernel)
		int64_t* offset = reinterpret_cast<int64_t*>(alloca(order * sizeof(int64_t)));
		for (uint64_t i = 0; i < order; i++)
		{
			offset[i] = static_cast<int64_t>(kernel.getSize(i) / 2);
		}

		// For each element of the original tensor
		int64_t* offsetedIndices = reinterpret_cast<int64_t*>(alloca(order * sizeof(int64_t)));
		for (const TensorPosition& pos : *this)
		{
			TValue value = _zero;

			// For each element of the kernel
			for (const TensorPosition& kernelPos : kernel)
			{
				bool setToZero = false;

				int64_t* itOffsetedIndices = offsetedIndices;
				const uint64_t* itShape = shape;
				const int64_t* itOffset = offset;
				const uint64_t* itIndices = pos.indices;
				const uint64_t* itKernelIndices = kernelPos.indices;

				// Compute the corresponding indices to poll
				for (uint64_t k = 0; k < order; ++k, ++itKernelIndices, ++itOffsetedIndices, ++itShape, ++itOffset, ++itIndices)
				{
					*itOffsetedIndices = static_cast<int64_t>(*itIndices) + *itOffset - static_cast<int64_t>(*itKernelIndices);

					switch (method)
					{
						case ConvolutionMethod::Zero:
							setToZero = (*itOffsetedIndices < 0 || *itOffsetedIndices >= *itShape);
							break;
						case ConvolutionMethod::Continuous:
							*itOffsetedIndices = std::clamp<int64_t>(*itOffsetedIndices, 0, *itShape - 1);
							break;
						case ConvolutionMethod::Periodic:
							*itOffsetedIndices = (*itOffsetedIndices + *itShape) % *itShape;
							break;
						default:
							assert(false);
					}

					if (setToZero)
					{
						break;
					}
				}

				// Add the product to the result
				if (!setToZero)
				{
					value += tensor->get(reinterpret_cast<uint64_t*>(offsetedIndices)) * kernel.get(kernelPos.indices);
				}
			}

			set(pos.indices, value);
		}

		delete tensor;

		return *this;
	}

	namespace _scp
	{
		template<typename T> struct IsComplex { static constexpr bool value = false; };
		template<typename T> struct IsComplex<std::complex<T>> { static constexpr bool value = true; };

		template<typename TValue>
		void cooleyTukey(TensorBase<TValue>& a, const std::vector<std::vector<TValue>>& exponentials, const uint64_t* shape, const uint64_t* offset, uint64_t* stride)
		{
			const uint64_t order = a.getOrder();

			// Directly exit if it is a single variable
			
			{
				uint64_t i = 0;
				for (; i < order; ++i)
				{
					if (shape[i] != 1)
					{
						break;
					}
				}

				if (i == order)
				{
					return;
				}
			}

			// Useful variables

			uint64_t* cellShape = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			for (uint64_t i = 0; i < order; ++i)
			{
				cellShape[i] = shape[i];
				for (uint64_t j = 2; j < shape[i]; ++j)
				{
					if (shape[i] % j == 0)
					{
						cellShape[i] = j;
						break;
					}
				}
			}

			uint64_t* subShape = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			for (uint64_t i = 0; i < order; ++i)
			{
				subShape[i] = shape[i] / cellShape[i];
			}

			uint64_t* subOffset = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			for (uint64_t i = 0; i < order; ++i)
			{
				subOffset[i] = offset[i] * cellShape[i];
			}

			// Compute fft of each sub tensor

			DynTensorSweep cellSweep(order, cellShape);
			for (const TensorPosition& aCellPos : cellSweep)
			{
				for (uint64_t i = 0; i < order; ++i)
				{
					stride[i] += aCellPos.indices[i] * offset[i];
				}

				cooleyTukey(a, exponentials, subShape, subOffset, stride);

				for (uint64_t i = 0; i < order; ++i)
				{
					stride[i] -= aCellPos.indices[i] * offset[i];
				}
			}

			// Store the coefficients calculated before merging them

			DynTensor<TValue> tmp(order, shape);
			uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			for (const TensorPosition& pos : tmp)
			{
				for (uint64_t i = 0; i < order; ++i)
				{
					indices[i] = stride[i] + pos.indices[i] * offset[i];
				}

				tmp.set(pos.indices, a.get(indices));
			}

			// Iterate over a sub tensor and merge each cell asociated

			uint64_t* aCellIndices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			uint64_t* tmpCellIndices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			uint64_t* aIndices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			uint64_t* tmpIndices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			DynTensorSweep subSweep(order, subShape);
			for (const TensorPosition& subPos : subSweep)
			{
				// Compute the position of the corner of the cell on 'a' and on 'tmp'

				for (uint64_t i = 0; i < order; ++i)
				{
					aCellIndices[i] = stride[i] + subPos.indices[i] * offset[i];
					tmpCellIndices[i] = subPos.indices[i] * cellShape[i];
				}

				// Merge the elements of the cell of 'tmp' into the elements of the cell of 'a'

				for (const TensorPosition& aCellPos : cellSweep)
				{
					for (uint64_t i = 0; i < order; ++i)
					{
						aIndices[i] = aCellIndices[i] + aCellPos.indices[i] * subShape[i] * offset[i];
					}

					TValue value = 0;

					for (const TensorPosition& tmpCellPos : cellSweep)
					{
						for (uint64_t i = 0; i < order; ++i)
						{
							tmpIndices[i] = tmpCellIndices[i] + tmpCellPos.indices[i];
						}

						TValue factor = 1;
						for (uint64_t i = 0; i < order; ++i)
						{
							factor *= exponentials[i][((subPos.indices[i] * tmpCellPos.indices[i]) % shape[i]) * offset[i]];
							factor *= exponentials[i][((tmpCellPos.indices[i] * aCellPos.indices[i]) % cellShape[i]) * subShape[i] * offset[i]];
						}

						value += factor * tmp.get(tmpIndices);
					}

					a.set(aIndices, value);
				}
			}
		}
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::fft()
	{
		if constexpr (_scp::IsComplex<TValue>::value)
		{
			const uint64_t order = getOrder();
			const uint64_t* shape = getShape();

			uint64_t* stride = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			std::fill(stride, stride + order, 0);

			uint64_t* offset = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			std::fill<uint64_t*, uint64_t>(offset, offset + order, 1);

			std::vector<std::vector<TValue>> exponentials(order);
			for (uint64_t i = 0; i < order; ++i)
			{
				exponentials[i].resize(shape[i]);
				for (uint64_t j = 0; j < shape[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, -2 * std::numbers::pi * j / shape[i]));
				}
			}

			_scp::cooleyTukey(*this, exponentials, shape, offset, stride);
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::ifft()
	{
		if constexpr (_scp::IsComplex<TValue>::value)
		{
			const uint64_t order = getOrder();
			const uint64_t* shape = getShape();

			uint64_t* stride = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			std::fill(stride, stride + order, 0);

			uint64_t* offset = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
			std::fill<uint64_t*, uint64_t>(offset, offset + order, 1);

			std::vector<std::vector<TValue>> exponentials(order);
			for (uint64_t i = 0; i < order; ++i)
			{
				exponentials[i].resize(shape[i]);
				for (uint64_t j = 0; j < shape[i]; ++j)
				{
					exponentials[i][j] = std::exp(TValue(0, 2 * std::numbers::pi * j / shape[i]));
				}
			}

			_scp::cooleyTukey(*this, exponentials, shape, offset, stride);

			*this /= getTotalLength();
		}
		else
		{
			assert(false);
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::negate()
	{
		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, -get(pos.indices));
		}

		return *this;
	}

	template<typename TValue>
	constexpr TValue TensorBase<TValue>::dotProduct(const TensorBase<TValue>& tensor) const
	{
		assert(getOrder() == tensor.getOrder());
		assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

		TValue result = _zero;
		for (const TensorPosition& pos : *this)
		{
			result += get(pos.indices) * tensor.get(pos.indices);
		}

		return result;
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::tensorContraction(uint64_t i, uint64_t j, TensorBase<TValue>& result) const
	{
		const uint64_t order = getOrder();
		const uint64_t* shape = getShape();
		const uint64_t resultOrder = result.getOrder();
		const uint64_t* resultShape = result.getShape();

		assert(order > 2);
		assert(resultOrder == order - 2);
		assert(i != j);
		assert(shape[i] == shape[j]);

		if (i > j)
		{
			std::swap(i, j);
		}

		assert(i == 0 || std::equal(shape, shape + i, resultShape));
		assert(j == i + 1 || std::equal(shape + i + 1, shape + j, resultShape + i));
		assert(j == order - 1 || std::equal(shape + j + 1, shape + order, resultShape + j - 1));

		uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
		for (const TensorPosition& pos : result)
		{
			if (i != 0)
			{
				std::copy(pos.indices, pos.indices + i, indices);
			}
			if (i != j - 1)
			{
				std::copy(pos.indices + i, pos.indices + j - 1, indices + i + 1);
			}
			if (j != order - 1)
			{
				std::copy(pos.indices + j - 1, pos.indices + resultOrder, indices + j + 1);
			}

			TValue value = 0;
			for (uint64_t k = 0; k < shape[i]; ++k)
			{
				indices[i] = k;
				indices[j] = k;
				value += get(indices);
			}

			result.set(pos.indices, value);
		}
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const
	{
		const uint64_t order = getOrder();
		const uint64_t tensorOrder = tensor.getOrder();
		const uint64_t resultOrder = result.getOrder();
		const uint64_t* shape = getShape();
		const uint64_t* tensorShape = tensor.getShape();
		const uint64_t* resultShape = result.getShape();

		assert(order + tensorOrder == resultOrder);
		assert(std::equal(shape, shape + order, resultShape) && std::equal(tensorShape, tensorShape + tensorOrder, resultShape + order));

		for (const TensorPosition& pos : result)
		{
			result.set(pos.indices, get(pos.indices) * tensor.get(pos.indices + order));
		}
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::contractedTensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const
	{
		const uint64_t order = getOrder();
		const uint64_t* shape = getShape();
		const uint64_t tensorOrder = tensor.getOrder();
		const uint64_t* tensorShape = tensor.getShape();
		const uint64_t resultOrder = result.getOrder();
		const uint64_t* resultShape = result.getShape();

		assert(order + tensorOrder > 2);
		assert(shape[order - 1] == tensorShape[0]);

		assert(resultOrder == order + tensorOrder - 2);
		assert(order == 1 || std::equal(shape, shape + order - 1, resultShape));
		assert(tensorOrder == 1 || std::equal(tensorShape + 1, tensorShape + tensorOrder, resultShape + order - 1));

		uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(order * sizeof(uint64_t)));
		uint64_t* tensorIndices = reinterpret_cast<uint64_t*>(alloca(tensorOrder * sizeof(uint64_t)));
		for (const TensorPosition& pos : result)
		{
			std::copy(pos.indices, pos.indices + order - 1, indices);
			std::copy(pos.indices + order - 1, pos.indices + resultOrder, tensorIndices + 1);

			TValue value = 0;
			for (uint64_t k = 0; k < tensorShape[0]; ++k)
			{
				indices[order - 1] = k;
				tensorIndices[0] = k;

				value += get(indices) * tensor.get(tensorIndices);
			}

			result.set(pos.indices, value);
		}
	}

	namespace _scp
	{
		template<typename TValue>
		constexpr double lerp(const TensorBase<TValue>& tensor, const uint64_t* shapeRev, uint64_t* indicesRev, const double* coeffsRev, uint64_t nCoeffs)
		{
			if (nCoeffs == 0)
			{
				return tensor.get(indicesRev + 1);
			}

			const double x = _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			++(*indicesRev);
			const double y = *indicesRev == *shapeRev ? x : _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			--(*indicesRev);

			return *coeffsRev * y + (1.0 - *coeffsRev) * x;
		}

		template<typename TValue>
		constexpr double cerp(const TensorBase<TValue>& tensor, const uint64_t* shapeRev, uint64_t* indicesRev, const double* coeffsRev, uint64_t nCoeffs)
		{
			if (nCoeffs == 0)
			{
				return tensor.get(indicesRev + 1);
			}

			const double x1 = _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			--(*indicesRev);
			const double x0 = *indicesRev == UINT64_MAX ? x1 : _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			*indicesRev += 2;
			const double x2 = *indicesRev == *shapeRev ? x1 : _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
			double x3 = x2;
			if (*indicesRev != *shapeRev)
			{
				++(*indicesRev);
				x3 = *indicesRev == *shapeRev ? x2 : _scp::lerp(tensor, shapeRev - 1, indicesRev - 1, coeffsRev - 1, nCoeffs - 1);
				--(*indicesRev);
			}
			--(*indicesRev);

			const double a = -0.5*x0 + 1.5*x1 - 1.5*x2 + 0.5*x3;
			const double b =      x0 - 2.5*x1 + 2.0*x2 - 0.5*x3;
			const double c = -0.5*x0          + 0.5*x2;
			const double d =               x1;

			const double& t = *coeffsRev;

			return d + t*(c + t*(b + t*a));
		}
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::interpolation(TensorBase<TValue>& result, InterpolationMethod method) const
	{
		if constexpr (std::is_convertible<TValue, double>::value && std::is_convertible<double, TValue>::value)
		{
			const uint64_t order = getOrder();
			const uint64_t resultOrder = result.getOrder();
			const uint64_t* shape = getShape();
			const uint64_t* resultShape = result.getShape();

			assert(order == resultOrder);

			uint64_t* indices = reinterpret_cast<uint64_t*>(alloca(sizeof(uint64_t) * order));
			double* coeffs = reinterpret_cast<double*>(alloca(sizeof(double) * order));

			double* shapeRatio = reinterpret_cast<double*>(alloca(sizeof(double) * order));
			for (uint64_t i = 0; i < order; ++i)
			{
				shapeRatio[i] = static_cast<double>(shape[i] - 1) / (resultShape[i] - 1);
			}

			const uint64_t* shapeRev = shape + order - 1;
			uint64_t* indicesRev = indices + order - 1;
			const double* coeffsRev = coeffs + order - 1;

			for (const TensorPosition& pos : result)
			{
				for (uint64_t i = 0; i < order; ++i)
				{
					coeffs[i] = pos.indices[i] * shapeRatio[i];
					indices[i] = static_cast<uint64_t>(coeffs[i]);
					coeffs[i] -= indices[i];
				}

				switch (method)
				{
					case InterpolationMethod::Nearest:
					{
						result.set(pos.indices, get(indices));
						break;
					}
					case InterpolationMethod::Linear:
					{
						result.set(pos.indices, _scp::lerp(*this, shapeRev, indicesRev, coeffsRev, order));
						break;
					}
					case InterpolationMethod::Cubic:
					{
						result.set(pos.indices, _scp::cerp(*this, shapeRev, indicesRev, coeffsRev, order));
						break;
					}
				}
			}
		}
		else
		{
			assert(false);
		}
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::fill(const TValue& value)
	{
		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, value);
		}

		return *this;
	}

	template<typename TValue>
	constexpr TensorBase<TValue>& TensorBase<TValue>::apply(const std::function<TValue(const TValue&)>& function)
	{
		assert(function);

		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, function(get(pos.indices)));
		}

		return *this;
	}

	template<typename TValue>
	constexpr TValue TensorBase<TValue>::normSq() const
	{
		TValue result = _zero;
		for (const TensorPosition& pos : *this)
		{
			const TValue& x = get(pos.indices);
			result += x * x;
		}

		return result;
	}

	template<typename TValue>
	constexpr TValue TensorBase<TValue>::norm() const
	{
		return std::sqrt(normSq());
	}

	template<typename TValue>
	constexpr const TValue& TensorBase<TValue>::minElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;

		if constexpr (isTotallyOrdered)
		{
			auto comp = [this](const TensorPosition& posA, const TensorPosition& posB) -> bool
			{
				return get(posA.indices) < get(posB.indices);
			};

			return get(std::min_element(begin(*this), end(*this), comp)->indices);
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& TensorBase<TValue>::maxElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;

		if constexpr (isTotallyOrdered)
		{
			auto comp = [this](const TensorPosition& posA, const TensorPosition& posB) -> bool
			{
				return get(posA.indices) < get(posB.indices);
			};

			return get(std::max_element(begin(*this), end(*this), comp)->indices);
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::copyFrom(const TensorBase<TValue>& tensor)
	{
		const uint64_t order = getOrder();
		const uint64_t tensorOrder = tensor.getOrder();
		const uint64_t* const shape = getShape();
		const uint64_t* const tensorShape = tensor.getShape();

		if (order != tensorOrder || !std::equal(shape, shape + order, tensorShape))
		{
			destroy();
			create(tensorOrder, tensorShape);
		}

		for (const TensorPosition& pos : *this)
		{
			set(pos.indices, tensor.get(pos.indices));
		}
	}

	template<typename TValue>
	constexpr void TensorBase<TValue>::moveFrom(TensorBase<TValue>&& tensor)
	{
		copyFrom(tensor);
	}


	template<TensorConcept TTensor>
	constexpr TTensor operator+(const TTensor& a, const TTensor& b)
	{
		TTensor c(a);
		c += b;
		return c;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& a, const TTensor& b)
	{
		a += b;
		return std::move(a);
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(const TTensor& a, TTensor&& b)
	{
		b += a;
		return std::move(b);
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& a, TTensor&& b)
	{
		a += b;
		return std::move(a);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator-(const TTensor& a, const TTensor& b)
	{
		TTensor c(a);
		c -= b;
		return c;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& a, const TTensor& b)
	{
		a -= b;
		return std::move(a);
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(const TTensor& a, TTensor&& b)
	{
		b -= a;
		return -std::move(b);
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& a, TTensor&& b)
	{
		a -= b;
		return std::move(a);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator*(const TTensor& tensor, const typename TTensor::ValueType& value)
	{
		TTensor result(tensor);
		result *= value;
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator*(TTensor&& tensor, const typename TTensor::ValueType& value)
	{
		tensor *= value;
		return std::move(tensor);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator*(const typename TTensor::ValueType& value, const TTensor& tensor)
	{
		TTensor result(tensor);
		result *= value;
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator*(const typename TTensor::ValueType& value, TTensor&& tensor)
	{
		tensor *= value;
		return std::move(tensor);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator/(const TTensor& tensor, const typename TTensor::ValueType& value)
	{
		TTensor result(tensor);
		result /= value;
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator/(TTensor&& tensor, const typename TTensor::ValueType& value)
	{
		tensor /= value;
		return std::move(tensor);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator-(const TTensor& tensor)
	{
		TTensor result(tensor);
		result.negate();
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator-(TTensor&& tensor)
	{
		tensor.negate();
		return std::move(tensor);
	}

	template<TensorConcept TTensor>
	constexpr TTensor operator+(const TTensor& tensor)
	{
		return tensor;
	}

	template<TensorConcept TTensor>
	constexpr TTensor&& operator+(TTensor&& tensor)
	{
		return std::move(tensor);
	}

	template<TensorConcept TTensor>
	constexpr TTensor hadamardProduct(const TTensor& a, const TTensor& b)
	{
		TTensor c(a);
		c.hadamardProduct(b);
		return c;
	}

	template<TensorConcept TTensor>
	constexpr TTensor convolution(const TTensor& tensor, const TensorBase<typename TTensor::ValueType>& kernel, ConvolutionMethod method)
	{
		TTensor result(tensor);
		result.convolution(kernel, method);
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor fft(const TTensor& tensor)
	{
		TTensor result(tensor);
		result.fft();
		return result;
	}

	template<TensorConcept TTensor>
	constexpr TTensor ifft(const TTensor& tensor)
	{
		TTensor result(tensor);
		result.ifft();
		return result;
	}

	template<typename TValue>
	constexpr typename TValue dotProduct(const TensorBase<TValue>& a, const TensorBase<TValue>& b)
	{
		return a.dotProduct(b);
	}
}
