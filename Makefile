GCC=clang++
CFLAGS=-Wall -g -O3
LOADLIBS=-lboost_system -lboost_filesystem

clean:
	rm -rf *.o *.out

mepredict : MEModel.o MEFeature.o main.cpp
	$(GCC) $(CFLAGS) -o mepredict MEModel.o MEFeature.o main.cpp $(LOADLIBS) 

nextword_test : MEModel.o MEFeature.o nextword_test.cpp
	$(GCC) $(CFLAGS) -o nextword_test MEModel.o MEFeature.o nextword_test.cpp 

MEModel.o : MEModel.hpp MEModel.cpp MEFeature.hpp 
	$(GCC) $(CFLAGS) -c MEModel.cpp

MEFeature.o : MEFeature.hpp MEFeature.cpp
	$(GCC) $(CFLAGS) -c MEFeature.cpp
