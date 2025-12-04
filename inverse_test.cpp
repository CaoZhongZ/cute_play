#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main() {
  Layout<
    Shape< Shape<_256, _128>, _1, _2 >,
    Stride< Stride<_128, _1>, _0, _32768> > layout {};

  auto r_inv = right_inverse(layout);

  std::cout << "Layout: " << layout << std::endl;
  std::cout << "Right inverse: " << right_inverse(layout) << std::endl;
  std::cout << "Left inverse: " << left_inverse(layout) << std::endl;

  Layout<Shape<_256, _128>, Stride<_1024, _1>> partial {};
  std::cout << "Partial: " << partial << std::endl;
  auto p_inv = right_inverse(partial);
  std::cout << "Rignt inverse of Partial: " << p_inv << std::endl;

  Layout<
    Shape< Shape<_256, _128>, _1, _2 >,
    Stride< Stride<_256, _1>, _0, _128> > horizontal {};
  auto h_inv = right_inverse(horizontal);

  std::cout << "Horizontal stack: " << horizontal << std::endl;
  std::cout << "Right inverse of horizontal stack: " << h_inv << std::endl;


  return 0;
}
