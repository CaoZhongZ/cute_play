#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto my_shape = make_shape(Int<512>{}, Int<256>{}, Int<128>{});

  std::cout << "Shape of the identity layout: " << my_shape <<std::endl;

  return 0;
}
