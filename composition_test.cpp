#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto a = Layout<Shape <Int<20>>, Stride<_2>>{};
  auto b = Layout<Shape <_5, _4>, Stride<_4, _1>>{};
  auto result = composition(a, b);
  // Identical to

  std::cout << "Result layout: " << result <<std::endl;

  auto c = Layout<Shape <_2, _4>, Stride<_4, _1>>{};
  auto d = Layout<Shape <_8, _8>, Stride<_8, _1>>{};

  auto s = make_identity_layout(Shape <_2, _4> {});

  std::cout << "Result layout: " << composition(d, c) <<std::endl;
  std::cout << d << " x "<< s <<" = " << composition(d, s) <<std::endl;

  {
    Layout a = make_layout(make_shape (Int<10>{}, Int<2>{}),
                           make_stride(Int<16>{}, Int<4>{}));
    Layout b = make_layout(make_shape (Int< 5>{}, Int<4>{}),
                           make_stride(Int< 1>{}, Int<5>{}));
    Layout c = composition(a, b);
    std::cout << c << std::endl;
  }

  {
    // (12,(4,8)):(59,(13,1))
    auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                         make_stride(59,make_stride(13,1)));
    // <3:4, 8:2>
    auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                           Layout<_8,_2>{}); // Apply 8:2 to mode-1
    
    // (_3,(2,4)):(236,(26,1))
    auto result = composition(a, tiler);
    // Identical to
    auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                              composition(layout<1>(a), get<1>(tiler)));

    std::cout << result << std::endl;
  }

  return 0;
}
