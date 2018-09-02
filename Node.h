#pragma once
#include <vector>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\functional.h>
#include <thrust\transform.h>
#include <thrust\reduce.h>
#include <thrust\for_each.h>
#include <thrust\copy.h>
#include <thrust\device_ptr.h>
#define N 4.0
#define d 0.85

class Node {

public:

	thrust::host_vector<Node*> outlinks;
	thrust::host_vector<Node*> inlinks;
	int niz[1] = { 3 };

	Node();
	~Node() {};
	float PRinLastIter;
	float PRinCurrentIter;
	int sizeOfOutLinks;
	std::string url;
	double getCurrentPR();
	double getPRinLastIter();
	void setPR(float);
	void setUrl(std::string);
	std::string getUrl();

	void pushNewOutGoingLink(Node*);
	void pushNewInGoingLink(Node*);
	
	thrust::host_vector<Node*> getInGoingLinks();
	thrust::host_vector<Node*> getOutGoingLinks();
	void printNodeInfo();
	void updateLastPR();

	Node* inGoingLinksPointer();

};