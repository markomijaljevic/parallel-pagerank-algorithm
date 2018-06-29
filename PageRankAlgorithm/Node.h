#pragma once
#include <vector>
#define N 2.0
#define d 0.85

class Node {
	float PRinLastIter;
	float PRinCurrentIter;
	std::string url;
	std::vector<Node*> outgoinglinks;
	std::vector<Node*> ingoinglinks;
public:
	Node();
	~Node() {};

	double getCurrentPR();
	double getPRinLastIter();
	void setPR(float);
	void setUrl(std::string);
	std::string getUrl();
	void pushNewOutGoingLink(Node*);
	void pushNewInGoingLink(Node*);
	std::vector<Node*> getInGoingLinks();
	std::vector<Node*> getOutGoingLinks();
	void printNodeInfo();
	void updateLastPR();
};