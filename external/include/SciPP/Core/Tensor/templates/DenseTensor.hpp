#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/DenseTensor.hpp>

namespace scp
{
	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::operator+=(const TensorBase<TValue>& tensor)
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			assert(getOrder() == tensor.getOrder());
			assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

			TValue* itA = getData();
			const TValue* itB = denseTensor->getData();
			const TValue* const itAEnd = itA + this->getTotalLength();
			for (; itA != itAEnd; ++itA, ++itB)
			{
				*itA += *itB;
			}
		}
		else
		{
			TensorBase<TValue>::operator+=(tensor);
		}

		return *this;
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::operator-=(const TensorBase<TValue>& tensor)
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			assert(getOrder() == tensor.getOrder());
			assert(std::equal(getShape(), getShape() + getOrder(), tensor.getShape()));

			TValue* itA = getData();
			const TValue* itB = denseTensor->getData();
			const TValue* const itAEnd = itA + this->getTotalLength();
			for (; itA != itAEnd; ++itA, ++itB)
			{
				*itA -= *itB;
			}
		}
		else
		{
			TensorBase<TValue>::operator-=(tensor);
		}

		return *this;
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::operator*=(const TValue& value)
	{
		TValue* it = getData();
		const TValue* const itEnd = it + this->getTotalLength();
		for (; it != itEnd; ++it)
		{
			*it *= value;
		}

		return *this;
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::operator/=(const TValue& value)
	{
		TValue* it = getData();
		const TValue* const itEnd = it + this->getTotalLength();
		for (; it != itEnd; ++it)
		{
			*it /= value;
		}

		return *this;
	}

	template<typename TValue>
	constexpr bool DenseTensor<TValue>::operator==(const TensorBase<TValue>& tensor) const
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			const uint64_t order = getOrder();
			if (order != denseTensor->getOrder())
			{
				return false;
			}

			const uint64_t* shape = getShape();
			if (!std::equal(shape, shape + order, denseTensor->getShape()))
			{
				return false;
			}

			const TValue* itA = getData();
			const TValue* itB = denseTensor->getData();
			const TValue* const itAEnd = itA + this->getTotalLength();
			for (; itA != itAEnd; ++itA, ++itB)
			{
				if (*itA != *itB)
				{
					return false;
				}
			}

			return true;
		}
		else
		{
			return TensorBase<TValue>::operator==(tensor);
		}
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::hadamardProduct(const TensorBase<TValue>& tensor)
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			TValue* itA = getData();
			const TValue* itB = denseTensor->getData();
			const TValue* const itAEnd = itA + this->getTotalLength();
			for (; itA != itAEnd; ++itA, ++itB)
			{
				*itA *= *itB;
			}
		}
		else
		{
			TensorBase<TValue>::hadamardProduct(tensor);
		}

		return *this;
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::negate()
	{
		TValue* it = getData();
		const TValue* const itEnd = it + this->getTotalLength();
		for (; it != itEnd; ++it)
		{
			*it = -(*it);
		}

		return *this;
	}

	template<typename TValue>
	constexpr TValue DenseTensor<TValue>::dotProduct(const TensorBase<TValue>& tensor) const
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			const TValue* itA = getData();
			const TValue* itB = denseTensor->getData();
			const TValue* const itAEnd = itA + this->getTotalLength();

			TValue result = 0;
			for (; itA != itAEnd; ++itA, ++itB)
			{
				result += *itA * *itB;
			}

			return result;
		}
		else
		{
			return TensorBase<TValue>::dotProduct(tensor);
		}
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::tensorProduct(const TensorBase<TValue>& tensor, TensorBase<TValue>& result) const
	{
		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		DenseTensor<TValue>* denseResult = dynamic_cast<DenseTensor<TValue>*>(&result);
		if (denseTensor && denseResult) // TODO
		{
			const uint64_t order = getOrder();
			const uint64_t tensorOrder = tensor.getOrder();
			const uint64_t resultOrder = result.getOrder();
			const uint64_t* shape = getShape();
			const uint64_t* tensorShape = tensor.getShape();
			const uint64_t* resultShape = result.getShape();

			assert(order + tensorOrder == resultOrder);
			assert(std::equal(shape, shape + order, resultShape) && std::equal(tensorShape, tensorShape + tensorOrder, resultShape + order));

			TValue* itResult = denseResult->getData();
			const TValue* itA = getData();
			const TValue* const itAEnd = itA + this->getTotalLength();
			for (; itA != itAEnd; ++itA)
			{
				const TValue* itB = denseTensor->getData();
				const TValue* const itBEnd = itB + this->getTotalLength();
				for (; itB != itBEnd; ++itB, ++itResult)
				{
					*itResult = *itA * *itB;
				}
			}
		}
		else
		{
			TensorBase<TValue>::tensorProduct(tensor, result);
		}
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::fill(const TValue& value)
	{
		TValue* it = getData();
		std::fill(it, it + this->getTotalLength(), value);

		return *this;
	}

	template<typename TValue>
	constexpr DenseTensor<TValue>& DenseTensor<TValue>::apply(const std::function<TValue(const TValue&)>& function)
	{
		TValue* it = getData();
		std::transform(it, it + this->getTotalLength(), it, function);

		return *this;
	}

	template<typename TValue>
	constexpr TValue DenseTensor<TValue>::normSq() const
	{
		TValue result = 0;
		const TValue* it = getData();
		const TValue* const itEnd = it + this->getTotalLength();
		for (; it != itEnd; ++it)
		{
			result += *it * *it;
		}

		return result;
	}

	template<typename TValue>
	constexpr const TValue& DenseTensor<TValue>::minElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;
		if constexpr (isTotallyOrdered)
		{
			return *std::min_element(getData(), getData() + this->getTotalLength());
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& DenseTensor<TValue>::maxElement() const
	{
		constexpr bool isTotallyOrdered = std::totally_ordered<TValue>;
		if constexpr (isTotallyOrdered)
		{
			return *std::max_element(getData(), getData() + this->getTotalLength());
		}
		else
		{
			assert(false);
			return _zero;
		}
	}

	template<typename TValue>
	constexpr const TValue& DenseTensor<TValue>::get(uint64_t internalIndex) const
	{
		assert(internalIndex < getInternalLength());
		return *(getData() + internalIndex);
	}

	template<typename TValue>
	constexpr const TValue& DenseTensor<TValue>::get(const uint64_t* indices) const
	{
		return get(getInternalIndex(indices));
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::set(uint64_t internalIndex, const TValue& value)
	{
		assert(internalIndex < getInternalLength());
		*(getData() + internalIndex) = value;
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::set(const uint64_t* indices, const TValue& value)
	{
		set(getInternalIndex(indices), value);
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::getIndices(uint64_t internalIndex, uint64_t* indices) const
	{
		uint64_t factor = this->getTotalLength();

		uint64_t* itIndices = indices;
		const uint64_t* itShape = getShape();
		const uint64_t* const itShapeEnd = itShape + getOrder();

		for (; itShape != itShapeEnd; ++itIndices, ++itShape)
		{
			factor /= *itShape;
			const uint64_t index = internalIndex / factor;
			*itIndices = index;
			internalIndex -= index * factor;
		}
	}

	template<typename TValue>
	constexpr uint64_t DenseTensor<TValue>::getInternalIndex(const uint64_t* indices) const
	{
		const uint64_t* itIndices = indices;
		const uint64_t* itShape = getShape();
		const uint64_t* const itShapeEnd = itShape + getOrder();
		uint64_t index = 0;
		for (; itShape != itShapeEnd; ++itShape, ++itIndices)
		{
			assert(*itIndices < *itShape);
			index = index * *itShape + *itIndices;
		}

		return index;
	}

	template<typename TValue>
	constexpr uint64_t DenseTensor<TValue>::getInternalLength() const
	{
		return this->getTotalLength();
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::setInitialPosition(TensorIteratorBase* iterator, bool end) const
	{
		iterator->getInternalIndex() = 0;
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::incrementPosition(TensorIteratorBase* iterator) const
	{
		++iterator->getInternalIndex();
	}

	template<typename TValue>
	constexpr void DenseTensor<TValue>::copyFrom(const TensorBase<TValue>& tensor)
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

		const DenseTensor<TValue>* denseTensor = dynamic_cast<const DenseTensor<TValue>*>(&tensor);
		if (denseTensor)
		{
			TValue* it = getData();
			const TValue* itTensor = denseTensor->getData();
			const TValue* itTensorEnd = itTensor + this->getTotalLength();
			std::copy(itTensor, itTensorEnd, it);
		}
		else
		{
			TValue* it = getData();
			for (const TensorPosition& pos : tensor)
			{
				*it = tensor.get(pos.indices);
			}
		}
	}


	template<DenseTensorConcept TTensor>
	template<typename... Args>
	constexpr DenseMatrix<TTensor>::DenseMatrix(Args... args) : MatrixBase<TTensor>(std::forward<Args>(args)...)
	{
	}

	template<DenseTensorConcept TTensor>
	constexpr DenseMatrix<TTensor>::DenseMatrix(const TensorBase<typename DenseMatrix<TTensor>::ValueType>& tensor) : MatrixBase<TTensor>(tensor)
	{
	}

	template<DenseTensorConcept TTensor>
	constexpr void DenseMatrix<TTensor>::matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const
	{
		const DenseTensor<ValueType>* denseMatrix = dynamic_cast<const DenseTensor<ValueType>*>(&matrix);
		DenseTensor<ValueType>* denseResult = dynamic_cast<DenseTensor<ValueType>*>(&result);
		if (denseMatrix && denseResult) // TODO
		{
			assert(denseMatrix->getOrder() == 2);
			assert(denseMatrix->getSize(0) == this->getSize(1));
			assert(denseResult->getOrder() == 2);
			assert(denseResult->getSize(0) == this->getSize(0));
			assert(denseResult->getSize(1) == denseMatrix->getSize(1));

			const uint64_t m = this->getSize(0);
			const uint64_t n = this->getSize(1);
			const uint64_t p = matrix.getSize(1);

			ValueType* it = denseResult->getData();
			const ValueType* itA = this->getData();
			const ValueType* itB = denseMatrix->getData();

			for (uint64_t i = 0; i < m; ++i)
			{
				for (uint64_t j = 0; j < p; ++j)
				{
					*it = _zero;
					for (uint64_t k = 0; k < n; ++k)
					{
						++itA;
						itB += p;
						*it += *itA * *itB;
					}

					itB -= n * p - 1;
					itA -= n;

					++it;
				}

				itB -= p;
				itA += n;
			}
		}
		else
		{
			MatrixBase<TTensor>::matrixProduct(matrix, result);
		}
	}

	template<DenseTensorConcept TTensor>
	constexpr void DenseMatrix<TTensor>::vectorProduct(const TensorBase<ValueType>& vector, TensorBase<ValueType>& result) const
	{
		const DenseTensor<ValueType>* denseVector = dynamic_cast<const DenseTensor<ValueType>*>(&vector);
		DenseTensor<ValueType>* denseResult = dynamic_cast<DenseTensor<ValueType>*>(&result);
		if (denseVector && denseResult) // TODO
		{
			assert(denseVector->getOrder() == 1);
			assert(denseVector->getSize(0) == this->getSize(1));
			assert(denseResult->getOrder() == 1);
			assert(denseResult->getSize(0) == this->getSize(0));

			const uint64_t m = this->getSize(0);
			const uint64_t n = this->getSize(1);

			ValueType* it = denseResult->getData();
			const ValueType* itA = this->getData();
			const ValueType* itB = denseVector->getData();

			for (uint64_t i = 0; i < m; ++i, ++it)
			{
				*it = _zero;
				for (uint64_t j = 0; j < n; ++j, ++itA, ++itB)
				{
					*it += *itA * *itB;
				}
				itB -= n;
			}
		}
		else
		{
			MatrixBase<TTensor>::vectorProduct(vector, result);
		}
	}


	template<DenseTensorConcept TTensor>
	template<typename... Args>
	constexpr DenseVector<TTensor>::DenseVector(Args... args) : VectorBase<TTensor>(std::forward<Args>(args)...)
	{
	}

	template<DenseTensorConcept TTensor>
	constexpr DenseVector<TTensor>::DenseVector(const TensorBase<typename DenseVector<TTensor>::ValueType>& tensor) : VectorBase<TTensor>(tensor)
	{
	}

	template<DenseTensorConcept TTensor>
	constexpr void DenseVector<TTensor>::matrixProduct(const TensorBase<ValueType>& matrix, TensorBase<ValueType>& result) const
	{
		const DenseTensor<ValueType>* denseMatrix = dynamic_cast<const DenseTensor<ValueType>*>(&matrix);
		DenseTensor<ValueType>* denseResult = dynamic_cast<DenseTensor<ValueType>*>(&result);
		if (denseMatrix && denseResult) // TODO
		{
			assert(denseMatrix->getOrder() == 2);
			assert(denseMatrix->getSize(0) == this->getSize(0));
			assert(denseResult->getOrder() == 1);
			assert(denseResult->getSize(0) == denseMatrix->getSize(1));

			const uint64_t m = denseMatrix->getSize(0);
			const uint64_t n = denseMatrix->getSize(1);

			ValueType* it = denseResult->getData();
			const ValueType* itA = this->getData();
			const ValueType* itB = denseMatrix->getData();

			for (uint64_t j = 0; j < n; ++j, ++it)
			{
				*it = _zero;
				for (uint64_t i = 0; i < m; ++i, ++itA, itB += n)
				{
					*it += *itA * *itB;
				}

				itA -= m;
				itB -= m * n - 1;
			}
		}
		else
		{
			VectorBase<TTensor>::matrixProduct(matrix, result);
		}
	}
}
