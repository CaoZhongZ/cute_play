#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto layout1 = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
  auto result1 = coalesce(layout1);    // _12:_1
  std::cout << "Result layout: " << result1 <<std::endl;

  auto a = Layout<Shape <_1, Shape <_1,_6>, _1>,
                  Stride<_3, Stride<_6,_2>, _2>>{};
  auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
  // Identical to
  auto same_r = make_layout(coalesce(layout<0>(a)),
                            coalesce(layout<1>(a)));

  std::cout << "Result layout: " << result <<std::endl;

  auto another = coalesce(a, Step<_1, _1, _1>{});
  std::cout << "Result layout: " << another <<std::endl;

  return 0;
}
