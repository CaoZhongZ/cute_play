#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Create a 4x4 shape
  auto shape = make_shape(Int<1024>{}, Int<128>{});

  // Create an identity layout for this shape.
  // The library automatically deduces a row-major stride.
  auto identity_layout = make_identity_layout(shape);
  auto normal_layout = make_layout(shape);

  // Print the layout to visualize its mapping
  print(identity_layout);

  // Example indexing
  // Logical coordinate (1, 2)
  auto index = identity_layout(Int<1>{}, Int<2>{});
  auto n_index = normal_layout(_1{}, _2{});

  // This will result in index 1 * 4 + 2 = 6, which is row-major order.
  std::cout << "\nIndex for logical coordinate (1, 2): " << index << std::endl;
  std::cout << "Index for normal coordinate (1, 2): " << n_index << std::endl;

  std::cout << identity_layout.compose(Int<128>{}, Int<256>{}) << std::endl;

  return 0;
}
