CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -Icutlass/include -Icutlass/tools/util/include -Iinclude

all : layout_test

clean:
	rm -f layout_test
