#include <iostream>
#include "Graph.h"
#include "Node.h"

int main() {
	
	Graph g;
	Node A, B, C, D,E;
	A.setUrl("www.A.com");
	B.setUrl("www.B.com");
	C.setUrl("www.C.com");
	D.setUrl("www.D.com");
	E.setUrl("www.E.com");

	A.pushNewOutGoingLink(&B);
	A.pushNewOutGoingLink(&C);
	B.pushNewOutGoingLink(&D);
	C.pushNewOutGoingLink(&A);
	C.pushNewOutGoingLink(&B);
	C.pushNewOutGoingLink(&D);
	D.pushNewOutGoingLink(&C);
	
	g.addNewNode(&A);
	g.addNewNode(&B);
	g.addNewNode(&C);
	g.addNewNode(&D);
	g.addNewNode(&E);

	g.PageRank();
	g.printGraphInfo();
	//E.printNodeInfo();

	//g.printGraphInfo();
	//std::cout << g.findNodeByUrl("www.A.com").getCurrentPR() << std::endl;

}
