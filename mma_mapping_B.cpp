
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(_16{}), make_stride(_16{}));

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(_2{}, _8{}), make_stride(_1{}, _2{}));   // (4,1) -> val_idx
  Layout TV_layout = make_layout(thr_layout, val_layout);

  Layout layout_mn = make_identity_layout(make_shape(_16{}, _16{}));

  auto mapping = composition(layout_mn, TV_layout);

  print(mapping); print("\n");
  print(mapping(1, 2)); print("\n");

  print_latex_tv(composition(layout_mn, TV_layout), make_tile(_16{}, _16{}));

}
