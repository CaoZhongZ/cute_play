#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto a = Layout<Shape <_2,Shape <_1,_6>>,
                  Stride<_1,Stride<_6,_2>>>{};
  auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
  // Identical to
  auto same_r = make_layout(coalesce(layout<0>(a)),
                            coalesce(layout<1>(a)));

  std::cout << "Result layout: " << same_r <<std::endl;

  return 0;
}
