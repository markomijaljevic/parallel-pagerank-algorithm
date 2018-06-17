#pragma once
#include <vector>
#include "Node.h"

class Graph {

	std::vector<Node*> Nodes;
public:
	Graph() {};
	~Graph() {};

	void addNewNode(Node*);
	size_t getNodeCount();
	Node* findNodeByUrl(std::string);
	void printGraphInfo();

	void PageRank();
};