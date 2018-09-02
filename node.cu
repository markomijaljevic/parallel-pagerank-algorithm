#include <iostream>
#include <string>
#include "Node.h"
#include <math.h>

Node::Node() {
	this->PRinCurrentIter = 1 / N;
	this->PRinLastIter = 1 / N;
	this->sizeOfOutLinks = 0;
}

void Node::setPR(float newPR) {
	this->PRinLastIter = this->PRinCurrentIter; //roundf( this->PRinCurrentIter * 1000) / 1000;  ==> round to 3 decimal places
	this->PRinCurrentIter = newPR; //roundf( this->PRinCurrentIter * 1000) / 1000;  ==> round to 3 decimal places

								   /*
								   Rounding to less decimal places , results with less iterations.
								   */
}

void Node::setUrl(std::string newUrl) {
	this->url = newUrl;
	//std::cout << this->url << std::endl;
}

double Node::getCurrentPR() {
	return this->PRinCurrentIter;
}

double Node::getPRinLastIter() {
	return this->PRinLastIter;
}

std::string Node::getUrl() {
	return this->url;
}

void Node::updateLastPR() {
	//std::cout << "Update..." << std::endl;
	this->PRinLastIter = this->PRinCurrentIter;

}


void Node::pushNewOutGoingLink(Node* node) {
	this->outlinks.push_back(node);
	node->pushNewInGoingLink(this);
}

void Node::pushNewInGoingLink(Node* node) {
	this->inlinks.push_back(node);
}

thrust::host_vector<Node*> Node::getInGoingLinks() {
	return this->inlinks;
}


thrust::host_vector<Node*> Node::getOutGoingLinks() {
	return this->outlinks;
}

void Node::printNodeInfo() {
	std::cout << "Url : " << this->getUrl() << std::endl;
	std::cout << "PR :" << this->getCurrentPR() << std::endl;
	std::cout << "In Going Links : " << this->inlinks.size() << std::endl;
	for (Node* node : this->inlinks) {
		std::cout << '\t' << "Url:" << node->getUrl() << std::endl;
		std::cout << '\t' << "PR:" << node->getCurrentPR() << std::endl;
		std::cout << '\t' << "OutLinks:" << node->sizeOfOutLinks << std::endl; // tu je problem
		std::cout << '\t' << "InLinks:" << node->getInGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Out Going Links : " << this->sizeOfOutLinks << std::endl;
	for (Node* node : this->outlinks) {
		std::cout << '\t' << "Url:" << node->getUrl() << std::endl;
		std::cout << '\t' << "PR:" << node->getCurrentPR() << std::endl;
		std::cout << '\t' << "OutLinks:" << node->getOutGoingLinks().size() << std::endl;
		std::cout << '\t' << "InLinks:" << node->getInGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << std::endl;
}