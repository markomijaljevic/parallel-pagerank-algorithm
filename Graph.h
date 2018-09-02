#pragma once
#include "Node.h"

class Graph {

public:

	//std::vector<Node*> Nodes;
	thrust::host_vector<Node*> Nodes;

	unsigned int iter_counter;
	float normalize_sum;

	Graph() { this->iter_counter = 0; }
	~Graph() {};

	void addNewNode(Node*);
	size_t getNodeCount();
	void printGraphInfo();
	
	Node* getInLinksNodes(thrust::host_vector<Node*>);

	Node* inLinks;


	void PageRank_GPU();
	void PageRank_CPU();

};