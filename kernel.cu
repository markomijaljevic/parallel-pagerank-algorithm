#include <iostream>
#include "Graph.h"
#include "Node.h"
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int main() {

	std::clock_t start;
	double duration;
	start = std::clock();

	Graph g;
	Node A, B, C, D, E, F, S1, S2, S3, S4, S5;
	A.setUrl("www.A.com");
	B.setUrl("www.B.com");
	C.setUrl("www.C.com");
	D.setUrl("www.D.com");
	E.setUrl("www.E.com");
	F.setUrl("www.F.com");
	S1.setUrl("www.S1.com");
	S2.setUrl("www.S2.com");
	S3.setUrl("www.S3.com");
	S4.setUrl("www.S4.com");
	S5.setUrl("www.S5.com");

	B.pushNewOutGoingLink(&C);
	C.pushNewOutGoingLink(&B);
	D.pushNewOutGoingLink(&A);
	D.pushNewOutGoingLink(&B);
	E.pushNewOutGoingLink(&B);
	E.pushNewOutGoingLink(&D);
	E.pushNewOutGoingLink(&F);
	F.pushNewOutGoingLink(&B);
	F.pushNewOutGoingLink(&E);

	S1.pushNewOutGoingLink(&B);
	S1.pushNewOutGoingLink(&E);
	S2.pushNewOutGoingLink(&B);
	S2.pushNewOutGoingLink(&E);
	S3.pushNewOutGoingLink(&B);
	S3.pushNewOutGoingLink(&E);
	S4.pushNewOutGoingLink(&E);
	S5.pushNewOutGoingLink(&E);

	g.addNewNode(&A);
	g.addNewNode(&B);
	g.addNewNode(&C);
	g.addNewNode(&D);
	g.addNewNode(&E);
	g.addNewNode(&F);
	g.addNewNode(&S1);
	g.addNewNode(&S2);
	g.addNewNode(&S3);
	g.addNewNode(&S4);
	g.addNewNode(&S5);

	
	g.PageRank_GPU();
	//g.PageRank_CPU();
	g.printGraphInfo();

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Vrijeme --> " << duration << '\n';
}