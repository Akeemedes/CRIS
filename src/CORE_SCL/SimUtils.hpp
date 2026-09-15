#ifndef __SIMUTILSHPP_
#define __SIMUTILSHPP_

#include <vector>
#include <iostream>

#define ASSERT_WITH_MSG(condition, message) \
    if (!(condition)) { \
        std::cerr << "ASSERTION FAILED: " << (message) << "\n"; \
        std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n"; \
        std::abort(); \
    }


template <typename T1>
void state_copy(T1& x, std::vector<double>& x_old) {
	auto size = x.size();
	x_old.resize(size);
	for (std::size_t i = 0; i < size; ++i) x_old[i] = x[i];
}

template <typename Ty>
void make_independent(Ty& elem, std::size_t i) {
	elem.make_independent(i);
}
void make_independent(double elem, std::size_t) {

}

#endif
