#include <iostream>
#include <string>
#include "Node.h"


Node::Node() {
	this->PRinCurrentIter = 1 / N;
	this->PRinLastIter = 1 / N;
}

void Node::setPR(double newPR) {
	this->PRinLastIter = this->PRinCurrentIter;
	this->PRinCurrentIter = newPR;
}

void Node::setUrl(std::string newUrl) {
	this->url = newUrl;
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
	this->PRinLastIter = this->PRinCurrentIter;
}

void Node::pushNewOutGoingLink(Node* node) {
	this->outgoinglinks.push_back(node);
	node->pushNewInGoingLink(this);
}

void Node::pushNewInGoingLink(Node* node) {
	this->ingoinglinks.push_back(node);
}

std::vector<Node*> Node::getInGoingLinks() {
	return this->ingoinglinks;
}
std::vector<Node*> Node::getOutGoingLinks() {
	return this->outgoinglinks;
}

void Node::printNodeInfo() {
	std::cout << "Url : " << this->getUrl() << std::endl;
	std::cout << "PR :" << this->getCurrentPR() << std::endl;
	std::cout << "In Going Links :" << std::endl;
	for (Node* node : this->ingoinglinks) {
		std::cout << '\t' << "Url:" << node->getUrl() << std::endl;
		std::cout << '\t' << "PR:" << node->getCurrentPR() << std::endl;
		std::cout << '\t' << "OutLinks:" << node->getOutGoingLinks().size() << std::endl;
		std::cout << '\t' << "InLinks:" << node->getInGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Out Going Links :" << std::endl;
	for (Node* node : this->outgoinglinks) {
		std::cout << '\t' << "Url:" << node->getUrl() << std::endl;
		std::cout << '\t' << "PR:" << node->getCurrentPR() << std::endl;
		std::cout << '\t' << "OutLinks:" << node->getOutGoingLinks().size() << std::endl;
		std::cout << '\t' << "InLinks:" << node->getInGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << std::endl;
}