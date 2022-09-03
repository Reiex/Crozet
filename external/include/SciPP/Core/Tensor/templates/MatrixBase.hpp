#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2019-2022
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SciPP/Core/Tensor/MatrixBase.hpp>

namespace scp
{
	template<TensorConcept TTensor>
	template<typename... Args>
	constexpr MatrixBase<TTensor>::MatrixBase(Args... args) : TTensor(std::forward<Args>(args)...)
	{
		assert(this->getOrder() == 2);
	}

	template<TensorConcept TTensor>
	constexpr MatrixBase<TTensor>::MatrixBase(const TensorBase<typename MatrixBase<TTensor>::ValueType>& tensor) : TTensor(tensor)
	{
	}

	template<TensorConcept TTensor>
	constexpr TensorBase<typename MatrixBase<TTensor>::ValueType>& MatrixBase<TTensor>::inverse()
	{
		assert(this->getSize(0) == this->getSize(1));

		const uint64_t size = this->getSize(0);
		TensorBase<ValueType>* copy = this->clone();

		for (uint64_t j = 0; j < size; ++j)
		{
			if (copy->get({ j, j }) == _zero)
			{
				for (uint64_t i = j + 1; i < size; ++i)
				{
					if (copy->get({ i, j }) != _zero)
					{
						ValueType value = copy->get({ i, j });
						for (uint64_t k = 0; k < size; ++k)
						{
							this->set({ i, k }, this->get({ i, k }) / value);
							copy->set({ i, k }, copy->get({ i, k }) / value);
							this->set({ j, k }, this->get({ j, k }) + this->get({ i, k }));
							copy->set({ j, k }, copy->get({ j, k }) + copy->get({ i, k }));
						}
						break;
					}
				}

				if (copy->get({ j, j }) == _zero)
				{
					delete copy;
					throw std::runtime_error("The matrix cannot be inverted.");
				}
			}
			else
			{
				ValueType value = copy->get({ j, j });
				for (uint64_t k = 0; k < size; ++k)
				{
					this->set({ j, k }, this->get({ j, k }) / value);
					copy->set({ j, k }, copy->get({ j, k }) / value);
				}
			}

			for (uint64_t i = j + 1; i < size; ++i)
			{
				if (copy->get({ i, j }) != _zero)
				{
					ValueType value = copy->get({ i, j });
					for (uint64_t k = 0; k < size; ++k)
					{
						this->set({ i, k }, this->get({ i, k }) - this->get({ j, k }) * value);
						copy->set({ i, k }, copy->get({ i, k }) - copy->get({ j, k }) * value);
					}
				}
			}
		}

		for (uint64_t j = size - 1; j != UINT64_MAX; --j)
		{
			for (uint64_t i = j - 1; i != UINT64_MAX; --i)
			{
				if (copy->get({ i, j }) != _zero)
				{
					ValueType value = copy->get({ i, j });
					for (uint64_t k = 0; k < size; ++k)
					{
						this->set({ i, k }, this->get({ i, k }) - this->get({ j, k }) * value);
						copy->set({ i, k }, copy->get({ i, k }) - copy->get({ j, k }) * value);
					}
				}
			}
		}

		delete copy;

		return *this;
	}

	template<TensorConcept TTensor>
	constexpr void MatrixBase<TTensor>::transpose(TensorBase<typename MatrixBase<TTensor>::ValueType>& result) const
	{
		assert(result.getOrder() == 2);
		assert(result.getSize(0) == this->getSize(1));
		assert(result.getSize(1) == this->getSize(0));

		const uint64_t m = this->getSize(0);
		const uint64_t n = this->getSize(1);

		for (uint64_t i = 0; i < m; ++i)
		{
			for (uint64_t j = 0; j < n; ++j)
			{
				result.set({ j, i }, this->get({ i, j }));
			}
		}
	}

	template<TensorConcept TTensor>
	constexpr void MatrixBase<TTensor>::matrixProduct(const TensorBase<typename MatrixBase<TTensor>::ValueType>& matrix, TensorBase<typename MatrixBase<TTensor>::ValueType>& result) const
	{
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == this->getSize(1));
		assert(result.getOrder() == 2);
		assert(result.getSize(0) == this->getSize(0));
		assert(result.getSize(1) == matrix.getSize(1));

		const uint64_t m = this->getSize(0);
		const uint64_t n = this->getSize(1);
		const uint64_t p = matrix.getSize(1);

		for (uint64_t i = 0; i < m; ++i)
		{
			for (uint64_t j = 0; j < p; ++j)
			{
				ValueType value = _zero;
				for (uint64_t k = 0; k < n; ++k)
				{
					value += this->get({ i, k }) * matrix.get({ k, j });
				}
				result.set({ i, j }, value);
			}
		}
	}

	template<TensorConcept TTensor>
	constexpr void MatrixBase<TTensor>::vectorProduct(const TensorBase<typename MatrixBase<TTensor>::ValueType>& vector, TensorBase<typename MatrixBase<TTensor>::ValueType>& result) const
	{
		assert(vector.getOrder() == 1);
		assert(vector.getSize(0) == this->getSize(1));
		assert(result.getOrder() == 1);
		assert(result.getSize(0) == this->getSize(0));

		const uint64_t m = this->getSize(0);
		const uint64_t n = this->getSize(1);

		for (uint64_t i = 0; i < m; ++i)
		{
			ValueType value = _zero;
			for (uint64_t j = 0; j < n; ++j)
			{
				value += this->get({ i, j }) * vector.get(&j);
			}
			result.set(&i, value);
		}
	}

	template<TensorConcept TTensor>
	constexpr typename MatrixBase<TTensor>::ValueType MatrixBase<TTensor>::determinant() const
	{
		assert(this->getSize(0) == this->getSize(1));

		const uint64_t size = this->getSize(0);
		TensorBase<ValueType>* copy = this->clone();

		ValueType det = 1;
		for (uint64_t j = 0; j < size; ++j)
		{
			if (copy->get({ j, j }) == _zero)
			{
				for (uint64_t i = j + 1; i < size; ++i)
				{
					if (copy->get({ i, j }) != _zero)
					{
						for (uint64_t k = 0; k < size; ++k)
						{
							copy->set({ j, k }, copy->get({ j, k }) + copy->get({ i, k }));
						}
						break;
					}
				}

				if (copy->get({ j, j }) == _zero)
				{
					delete copy;
					return _zero;
				}
			}

			ValueType diagValue = copy->get({ j, j });
			for (uint64_t i = j + 1; i < size; ++i)
			{
				if (copy->get({ i, j }) != _zero)
				{
					ValueType value = copy->get({ i, j })/ diagValue;
					for (uint64_t k = 0; k < size; ++k)
					{
						copy->set({ i, k }, copy->get({ i, k }) - copy->get({ j, k }) * value);
					}
				}
			}

			det *= diagValue;
		}

		delete copy;

		return det;
	}


	template<TensorConcept TTensor>
	TensorIterator<2> begin(const MatrixBase<TTensor>& matrix)
	{
		return TensorIterator<2>(&matrix, false);
	}

	template<TensorConcept TTensor>
	TensorIterator<2> end(const MatrixBase<TTensor>& matrix)
	{
		return TensorIterator<2>(&matrix, true);
	}


	template<MatrixConcept TMatrix>
	constexpr TMatrix inverse(const TMatrix& matrix)
	{
		TMatrix result(matrix);
		result.inverse();
		return result;
	}


	template<TensorConcept TTensor>
	template<typename... Args>
	constexpr VectorBase<TTensor>::VectorBase(Args... args) : TTensor(std::forward<Args>(args)...)
	{
		assert(this->getOrder() == 1);
	}

	template<TensorConcept TTensor>
	constexpr VectorBase<TTensor>::VectorBase(const TensorBase<typename VectorBase<TTensor>::ValueType>& tensor) : TTensor(tensor)
	{
	}

	template<TensorConcept TTensor>
	constexpr TensorBase<typename VectorBase<TTensor>::ValueType>& VectorBase<TTensor>::crossProduct(const TensorBase<typename VectorBase<TTensor>::ValueType>& vector)
	{
		assert(this->getSize(0) == 3);
		assert(vector.getOrder() == 1);
		assert(vector.getSize(0) == 3);

		ValueType xA = this->get({ 0 });
		ValueType yA = this->get({ 1 });
		ValueType zA = this->get({ 2 });

		const ValueType& xB = vector.get({ 0 });
		const ValueType& yB = vector.get({ 1 });
		const ValueType& zB = vector.get({ 2 });

		this->set({ 0 }, yA*zB - zA*yB);
		this->set({ 1 }, zA*xB - xA*zB);
		this->set({ 2 }, xA*yB - yA*xB);

		return *this;
	}

	template<TensorConcept TTensor>
	constexpr void VectorBase<TTensor>::matrixProduct(const TensorBase<typename VectorBase<TTensor>::ValueType>& matrix, TensorBase<typename VectorBase<TTensor>::ValueType>& result) const
	{
		assert(matrix.getOrder() == 2);
		assert(matrix.getSize(0) == this->getSize(0));
		assert(result.getOrder() == 1);
		assert(result.getSize(0) == matrix.getSize(1));

		const uint64_t m = matrix.getSize(0);
		const uint64_t n = matrix.getSize(1);

		for (uint64_t j = 0; j < n; ++j)
		{
			ValueType value = _zero;
			for (uint64_t i = 0; i < m; ++i)
			{
				value += matrix.get({ i, j }) * this->get(&i);
			}
			result.set(&j, value);
		}
	}


	template<TensorConcept TTensor>
	TensorIterator<1> begin(const VectorBase<TTensor>& vector)
	{
		return TensorIterator<1>(&vector, false);
	}

	template<TensorConcept TTensor>
	TensorIterator<1> end(const VectorBase<TTensor>& vector)
	{
		return TensorIterator<1>(&vector, true);
	}


	template<VectorConcept TVector>
	constexpr TVector crossProduct(const TVector& a, const TVector& b)
	{
		TVector result(a);
		result.crossProduct(b);
		return result;
	}
}
