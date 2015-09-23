GCC=g++
CFLAGS=-Wall -g -O3

clean:
	rm -rf *.o

nextword_test : MEModel.o MEFeature.o nextword_test.cpp
	$(GCC) $(CFLAGS) -o nextword_test MEModel.o MEFeature.o nextword_test.cpp 

MEModel.o : MEModel.hpp MEModel.cpp MEFeature.hpp 
	$(GCC) $(CFLAGS) -c MEModel.cpp

MEFeature.o : MEFeature.hpp MEFeature.cpp
	$(GCC) $(CFLAGS) -c MEFeature.cpp
