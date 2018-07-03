#pragma once
#include <vector>
#include "Node.h"

class Graph {

	std::vector<Node*> Nodes;
	unsigned int iter_counter;
	float normalize_sum;
public:
	Graph() { this->iter_counter = 0; }
	~Graph() {};

	void addNewNode(Node*);
	size_t getNodeCount();
	Node* findNodeByUrl(std::string);
	void printGraphInfo();

	void PageRank();
};