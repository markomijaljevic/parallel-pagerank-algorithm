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
		std::cout << "PR: " << node->getCurrentPR() << std::endl;
		std::cout << "In Going Links: " << node->getInGoingLinks().size() << std::endl;
		std::cout << "Out Going Links: " << node->getOutGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
}

void Graph::PageRank() {
	int flag = 0;
	double TempPR=0;
	std::vector<Node*> Inlinks;

	while (true) {
		if (flag == 1000)// num of iter
			break;
		for (Node* node : this->Nodes) {
			Inlinks = node->getInGoingLinks();
			if (Inlinks.size() == 0)
				continue;

			for (Node* n : Inlinks) {
				TempPR += n->getPRinLastIter() / n->getOutGoingLinks().size();
				//std::cout << "Last PR -> " << n->getPRinLastIter() << "/" << "Out Links -> " << n->getOutGoingLinks().size() << " = " << TempPR << std::endl;
				//std::cout << "PR of -> " << node->getUrl() << std::endl;
				//td::cout << "URL : " << n->getUrl() << " ->LastPR: " << n->getPRinLastIter() << std::endl;
			}
			//std::cout << std::endl;
			node->setPR(TempPR);
			TempPR = 0;
		}
		for (Node* n : this->Nodes) {
			n->updateLastPR();
		}
		//std::cout << "---------------------------------------------------------------------" << std::endl;
		flag++;
	}
	
}