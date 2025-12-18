#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

int main() {
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto rowmajor_layout = make_layout(make_shape(256, 512), LayoutRight {});
  auto colmajor_layout = make_layout(make_shape(256, 512), LayoutLeft {});

  {
    auto l_tile = logical_divide(rowmajor_layout, block_shape);
    auto t_tile = tiled_divide(rowmajor_layout, block_shape);

    std::cout<<"Logical divide "<<block_shape<<" from "
      <<rowmajor_layout<<" is "<<l_tile<<std::endl;
    std::cout<<"Tile divide "<<block_shape<<" from "
      <<rowmajor_layout<<" is "<<t_tile<<std::endl;
  }

  std::cout<<std::endl;

  {
    auto l_tile = logical_divide(colmajor_layout, block_shape);
    auto t_tile = tiled_divide(colmajor_layout, block_shape);

    std::cout<<"Logical divide "<<block_shape<<" from "
      <<colmajor_layout<<" is "<<l_tile<<std::endl;
    std::cout<<"Tile divide "<<block_shape<<" from "
      <<colmajor_layout<<" is "<<t_tile<<std::endl;
  }

  return 0;
}
