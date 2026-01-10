
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

  print(composition(layout_mn, TV_layout)); print("\n");

  print_latex_tv(composition(layout_mn, TV_layout), make_tile(_8{}, _16{}));

}
