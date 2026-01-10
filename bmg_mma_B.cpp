
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Thread arrangement
  // Layout thr_layout = make_layout(make_shape(_16{}), make_stride(_16{}));
  Layout thr_layout = make_layout(make_shape(_16{}), make_stride(_1{}));

  // Value arrangement per thread
  // Layout val_layout = make_layout(make_shape(_2{}, _8{}), make_stride(_1{}, _2{}));
  Layout val_layout = make_layout(make_shape(_2{}, _8{}), make_stride(_16{}, _32{}));
  Layout TV_layout = make_layout(thr_layout, val_layout);

  Layout layout_kn = make_identity_layout(make_shape(_16{}, _16{}));

  auto tile = make_shape(make_shape(_8{}, _2{}), _16{});
  Layout Layout_vnni = make_identity_layout(tile);

  auto vnni_layout = make_layout(make_shape(_16{}, make_shape(_2{}, _8{})), make_stride(_2{}, make_stride(_1{}, _32{})));

  // auto mapping = composition(layout_kn, TV_layout);

  // print(mapping); print("\n");
  // print(mapping(1, 2)); print("\n");

  // print_latex_tv(composition(layout_kn, TV_layout), make_shape(_16{}, _16{}));
  print(TV_layout); print("\n");
  print(vnni_layout); print("\n");
  print_latex(composition(vnni_layout, TV_layout)); print("\n");
}
