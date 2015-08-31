GCC=g++
CFLAGS=-Wall

clean:
	@rm -rf *.o

MEModel: MEFeature.o

MEFeature.o : MEFeature.hpp MEFeature.cpp
	$(GCC) $(CFLAGS) -c MEFeature.cpp
