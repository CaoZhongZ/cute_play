CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -Icutlass/include -Icutlass/tools/util/include -Iinclude

all : coalesce_test composition_test complement_test tiling_test product_test

clean:
	rm -rf coalesce_test composition_test complement_test tiling_test *.dSYM product_test identity_test layout_algtest divide_test inverse_test 01_mma_sm100_test 01_mma_sm100_test 10_mma_sm100_nvf4_test sf_layout
