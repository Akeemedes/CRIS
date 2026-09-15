#ifndef __STATEVECTORHPP_
#define __STATEVECTORHPP_

#include <cstddef>
#include <cmath>
#include <vector>

template <typename FieldType>
struct ConstantState
{
	static constexpr bool isConst = true;
	ConstantState (FieldType state ): mState(std::move(state))
	{
	}
	void update_state(double) const 
	{

	}
	double operator [] (std::size_t idx) const
	{
		return mState[idx];
	}

	std::size_t size() const
	{
		return mState.size();
	}

	private:
		const FieldType mState;
		double time;
};


#endif // __STATEVECTORHPP_ included
