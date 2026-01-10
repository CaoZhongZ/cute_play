
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(_8{}, _2{}), make_stride(_16 {}, _1{}));  // (32,8) -> thr_idx

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(_2{}, _4{}), make_stride(_8{}, _2{}));   // (4,1) -> val_idx
  Layout TV_layout = make_layout(thr_layout, val_layout);

  Layout layout_mn = make_identity_layout(make_shape(_8{}, _16{}));
  Layout col_mn = make_layout(make_shape(_8{}, _16{}), LayoutLeft {});
  Layout row_mn = make_layout(make_shape(_8{}, _16{}), LayoutRight {});

  print(composition(layout_mn, TV_layout)); print("\n");
  print(composition(row_mn, col_mn)); print("\n");
  auto row_pattern = composition(row_mn, TV_layout);
  print(row_pattern); print("\n");
  // print(row_pattern(8, 1)); print("\n");

  // print_latex_tv(row_pattern, make_tile(_8{}, _16{}));

}
