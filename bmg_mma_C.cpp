
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(_16{}), make_stride(_8{}));

  // Value arrangement per thread
  Layout val_layout = make_layout(make_shape(_8{}), make_stride(_1{}));
  Layout TV_layout = make_layout(thr_layout, val_layout);

  Layout layout_mn = make_identity_layout(make_shape(_8{}, _16{}));

  auto mapping = composition(layout_mn, TV_layout);

  print_latex_tv(composition(layout_mn, TV_layout), make_tile(_8{}, _16{}));

}
