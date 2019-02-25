BOOST_ROOT := /usr/include/boost
BOOST_INC := ${BOOST_ROOT}/include
LDFLAGS+=`pkg-config opencv --cflags --libs opencv`
mnisten: src/main.cpp
	g++ -Wall -O2 -std=c++17 -lboost_filesystem -lboost_thread -lboost_system $(LDFLAGS) -Iinclude -o mnisten src/main.cpp

