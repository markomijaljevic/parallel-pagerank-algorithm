#include "Graph.h"
#include <iostream>
#include <string>

void Graph::addNewNode(Node* newNode) {
	this->Nodes.push_back(newNode);
}

size_t Graph::getNodeCount() {
	return this->Nodes.size();
}
 
Node* Graph::findNodeByUrl(std::string url) {
	std::vector<Node*>::iterator iter;
	for (iter = Nodes.begin(); iter != Nodes.end(); iter++) {
		if ((*iter)->getUrl() == url)
			return *iter;
	}
}

void Graph::printGraphInfo() {
	for (Node* node : this->Nodes) {
		std::cout << "Url: " << node->getUrl() << std::endl;
		std::cout << "PR: " << ( node->getCurrentPR() / this->normalize_sum ) << std::endl;
		std::cout << "In Going Links: " << node->getInGoingLinks().size() << std::endl;
		std::cout << "Out Going Links: " << node->getOutGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Number of iterations: " << this->iter_counter << std::endl;
	std::cout << std::endl;
}

void Graph::PageRank() {
	bool flag = false;
	float TempPR=0;
	std::vector<Node*> Inlinks;
	
	while (true) {
		if (flag)
			break;

		flag = true;
		this->normalize_sum = 0;

		for (Node* node : this->Nodes) {
			Inlinks = node->getInGoingLinks();

			for (Node* n : Inlinks) {
				TempPR += n->getPRinLastIter() / n->getOutGoingLinks().size();
				//std::cout << "Last PR -> " << n->getPRinLastIter() << "/" << "Out Links -> " << n->getOutGoingLinks().size() << " = " << TempPR << std::endl;
				//std::cout << "PR of -> " << node->getUrl() << std::endl;
				//std::cout << "URL : " << n->getUrl() << " ->LastPR: " << n->getPRinLastIter() << std::endl;
			}
			//std::cout << std::endl;
			TempPR = (1 - d) + d * TempPR;
			node->setPR(TempPR);
			TempPR = 0;
		}
		//std::cout << "Iteration : " << this->iter_counter << std::endl;
		for (Node* n : this->Nodes) {
			//std::cout << " Last : " << n->getPRinLastIter() << " Current : " <<  n->getCurrentPR() << std::endl;
			if (n->getPRinLastIter() != n->getCurrentPR())
				flag = false;

			n->updateLastPR();
			this->normalize_sum += n->getCurrentPR();
		}
		//std::cout << "---------------------------------------------------------------------" << std::endl;
		this->iter_counter++;
	}
	
}