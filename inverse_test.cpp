#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main() {
  Layout<
    Shape< Shape<_16, _2>, _8>,
    Stride< Stride<_8, _0>, _1> > layout {};

  auto r_inv = right_inverse(layout);

  std::cout << "Layout: " << layout << std::endl;
  std::cout << "Right inverse: " << right_inverse(layout) << std::endl;
  std::cout << "Left inverse: " << left_inverse(layout) << std::endl;
  std::cout << std::endl;

  {
    Layout<Shape<_256, _128>, Stride<_1024, _1>> partial {};
    std::cout << "Partial: " << partial << std::endl;
    auto p_inv = right_inverse(partial);
    std::cout << "Rignt inverse of Partial: " << p_inv << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  {
    Layout<Shape<_256, _128>, Stride<_128, _1>> full {};
    std::cout << "Full: " << full << std::endl;
    auto p_inv = right_inverse(full);
    std::cout << "Rignt inverse of Full: " << p_inv << std::endl;
    auto p_inv1 = right_inverse(full).with_shape(full.shape());
    std::cout << "Rignt inverse with_shape of Full: " << p_inv1 << std::endl;
  }

  std::cout << std::endl;

  {
    Layout<Shape<_256, _128>, Stride<_256, _2>> full {};
    std::cout << "Not working layout: " << full << std::endl;
    auto p_inv = right_inverse(full);
    std::cout << "Rignt inverse of NotWork: " << p_inv << std::endl;
    auto p_inv1 = right_inverse(full).with_shape(full.shape());
    std::cout << "Rignt inverse of NotWork and with_shape: " << p_inv1 << std::endl;
  }

  std::cout << std::endl;

  {
    Layout<Shape<_256, _128>, Stride<_1, _256>> full {};
    std::cout << "Col: " << full << std::endl;
    auto p_inv = right_inverse(full);
    std::cout << "Rignt inverse of Col: " << p_inv << std::endl;
    auto p_inv1 = right_inverse(full).with_shape(full.shape());
    std::cout << "Rignt inverse and with shape of Col: " << p_inv1 << std::endl;
  }

  std::cout << std::endl;

  Layout<
    Shape< Shape<_256, _128>, _1, _2 >,
    Stride< Stride<_256, _1>, _0, _128> > horizontal {};
  auto h_inv = right_inverse(horizontal);

  std::cout << "Horizontal stack: " << horizontal << std::endl;
  std::cout << "Right inverse of horizontal stack: " << h_inv << std::endl;


  return 0;
}
