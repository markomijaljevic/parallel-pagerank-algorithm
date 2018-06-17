#pragma once
#include <vector>
#define N  4.0

class Node {
	double PRinLastIter;
	double PRinCurrentIter;
	std::string url;
	std::vector<Node*> outgoinglinks;
	std::vector<Node*> ingoinglinks;
public:
	Node();
	~Node() {};

	double getCurrentPR();
	double getPRinLastIter();
	void setPR(double);
	void setUrl(std::string);
	std::string getUrl();
	void pushNewOutGoingLink(Node*);
	void pushNewInGoingLink(Node*);
	std::vector<Node*> getInGoingLinks();
	std::vector<Node*> getOutGoingLinks();
	void printNodeInfo();
	void updateLastPR();
};