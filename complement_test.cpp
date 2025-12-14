#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto a = Layout<Shape <_5, _8>,
                  Stride<_2, _2>>{};
  std::cout << "Layout is " << a <<std::endl;
  std::cout << "It's coshape is " << coshape(a) <<std::endl;
  std::cout << "It's cosize is " << cosize(a) << std::endl;

  auto b = Layout<Shape <_4>, Stride<_2>> {};
  std::cout << "Cosize of "<<b<<" is "<<cosize(b)<<std::endl;
  std::cout << "Complement "<<b<<" of 24 is "<< complement(b, Int<24>{}) << std::endl;

  auto c = Layout<Shape <_6>, Stride<_4>> {};
  std::cout << "Composition " << b <<" and "<< c <<" is "<<composition(b, c) <<std::endl;

  {
    auto a = Layout<Shape <_2, _2>, Stride<_1, _6>> {};
    constexpr int c = 24;
    std::cout << "Compliment "<< a <<" with "<< c <<" is "<<complement(a, c)<<std::endl;
  }

  {
    auto a = Layout<Shape <_2, _2>, Stride<_2, _6>> {};
    constexpr int c = 24;
    std::cout << "This case show algorithm is best effort, not complete" <<std::endl;
    std::cout << "Compliment "<< a <<" with "<< c <<" is "<<complement(a, c)<<std::endl;
  }

  return 0;
}
