#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto a = Layout<Shape <_5, _8>,
                  Stride<_2, _2>>{};
  std::cout << "Layout is " << a <<std::endl;
  std::cout << "It's coshape is " << coshape(a) <<std::endl;
  std::cout << "It's cosize is " << cosize(a) << std::endl;

  auto b = Layout<Shape <_4>, Stride<_1>> {};
  std::cout << "Complement is "<< complement(b, Int<14>{}) << std::endl;

  return 0;
}
