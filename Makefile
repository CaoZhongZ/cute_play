CXX=clang++
CXXFLAGS=-g3 -O0 -ggdb3 -fno-inline -fno-omit-frame-pointer --std=c++17 -Icutlass/include -Icutlass/tools/util/include -Iinclude

all : coalesce_test composition_test

clean:
	rm -f coalesce_test composition_test
