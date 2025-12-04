
#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main() {
  ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
  ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
  print(*citer_2);
}

