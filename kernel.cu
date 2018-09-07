#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <thrust\reduce.h>
#include <thrust\device_ptr.h>
#include <thrust\device_vector.h>

#define N 11.0
#define d 0.85

int global_index = 0; // zamijeniti s velicinom vektora èvorova

struct Node {
	std::string url;
	double PRInLastIter = 1 / N;
	double PRInCurrentIter = 1 / N;
	int inLinks[10];
	int outLinks[5]; // ako radi dinamièki alocirati velièinu
	int index = 0;
	int outLinksIndex = 0;
	int inLinksIndex = 0;
	int sizeOfInLinks = 0;
	int sizeOfOutLinks = 0;
	int iter;
	Node() { this->index = global_index; global_index++; }
	~Node() { global_index--; }

	void isLinkingTo(Node&);
	void pushInLink(int);
};

struct Graph {
	std::vector<Node> nodes;
	int iterCounter = 0;
	double normalize_sum;
	void addNode(Node);
	void pagerank_gpu();
	void printGraphInfo();
};

void Node::isLinkingTo(Node& node) {
	this->outLinks[this->outLinksIndex]= node.index;
	this->outLinksIndex++;
	node.pushInLink(this->index);
}

void Node::pushInLink(int indexOfNode) {
	this->inLinks[this->inLinksIndex] = indexOfNode;
	this->inLinksIndex++;
}

void Graph::addNode(Node node) {
	node.sizeOfInLinks = node.inLinksIndex + 1;
	node.sizeOfOutLinks = node.outLinksIndex + 1;
	this->nodes.push_back(node);
}

void Graph::printGraphInfo() {
	for (Node node : this->nodes) {
		std::cout << "Url: " << node.url << std::endl;
		std::cout << "PR: " << (node.PRInCurrentIter / this->normalize_sum ) << std::endl;
		std::cout << "In Going Links: " << node.inLinksIndex + 1<< std::endl;
		std::cout << "Out Going Links: " << node.outLinksIndex + 1 << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Number of iterations: " << this->iterCounter << std::endl;
	std::cout << std::endl;
}

__global__ void calculatePageRank(Node* nodes,double* normSum, int* iterCounter) {

	int index = threadIdx.x;
	double PR = 0;
	int j;
	*iterCounter = -1;
	
	while (true){

		*iterCounter +=1;

		for (int i = 0; i < nodes[index].sizeOfInLinks - 1; i++) {
			j = nodes[index].inLinks[i];
			PR += ( nodes[j].PRInCurrentIter / ( nodes[j].sizeOfOutLinks - 1 ));
		}

		PR = (1 - d) + d * PR;

		nodes[index].PRInLastIter = nodes[index].PRInCurrentIter;
		nodes[index].PRInCurrentIter = PR;

		PR = 0;

		if (nodes[index].PRInLastIter != nodes[index].PRInCurrentIter)
			continue;

		normSum[index] = nodes[index].PRInCurrentIter;
		break;
	}
}

void Graph::pagerank_gpu() {

	Node* devNodes;
	int sizeOfNodes = this->nodes.size();
	int i = 0;
	cudaError_t cudaStatus;
	double* devNormSum;
	int* devIterCounter;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
		return;
	}

	cudaStatus = cudaMalloc((void**)&devNodes, sizeOfNodes * sizeof(Node));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed" << std::endl;
		return;
	}

	cudaStatus = cudaMalloc((void**)&devNormSum, sizeOfNodes * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed" << std::endl;
		return;
	}

	cudaStatus = cudaMalloc((void**)&devIterCounter, sizeOfNodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed" << std::endl;
		return;
	}

	cudaStatus = cudaMemcpy(devNodes, &this->nodes[0], sizeOfNodes * sizeof(Node), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed" << std::endl;
		return;
	}


	calculatePageRank << < 1, sizeOfNodes >> > (devNodes, devNormSum, devIterCounter);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "calculatePR launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;;
		return;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << "after launching calculatePR!" << std::endl;
		return;
	}

	cudaStatus = cudaMemcpy(&this->nodes[0], devNodes, sizeOfNodes * sizeof(Node), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed" << std::endl;
		return;
	}

	cudaStatus = cudaMemcpy(&this->iterCounter, devIterCounter, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed" << std::endl;
		return;
	}

	thrust::device_ptr<double> dev_ptr_sum(devNormSum);
	thrust::device_vector<double> norm_sum(dev_ptr_sum, dev_ptr_sum + sizeOfNodes);

	this->normalize_sum = thrust::reduce(norm_sum.begin(), norm_sum.end(), (float)0, thrust::plus<double>());

	cudaFree(devNodes);
	cudaFree(devNormSum);
	cudaFree(devIterCounter);
}

int main(void) {

	Graph g;
	Node A, B, C, D, E, F, S1, S2, S3, S4, S5;
	A.url = "wwww.a.com";
	B.url = "wwww.b.com";
	C.url = "wwww.c.com";
	D.url = "wwww.d.com";
	E.url = "wwww.e.com";
	F.url = "wwww.f.com";

	B.isLinkingTo(C);
	D.isLinkingTo(A);
	D.isLinkingTo(B);
	E.isLinkingTo(B);
	E.isLinkingTo(D);
	E.isLinkingTo(F);
	F.isLinkingTo(B);
	F.isLinkingTo(E);
	C.isLinkingTo(B);

	S1.isLinkingTo(B);
	S1.isLinkingTo(E);
	S2.isLinkingTo(B);
	S2.isLinkingTo(E);
	S3.isLinkingTo(B);
	S3.isLinkingTo(E);
	S4.isLinkingTo(E);
	S5.isLinkingTo(E);

	g.addNode(A);
	g.addNode(B);
	g.addNode(C);
	g.addNode(D);
	g.addNode(E);
	g.addNode(F);
	g.addNode(S1);
	g.addNode(S2);
	g.addNode(S3);
	g.addNode(S4);
	g.addNode(S5);
	
	g.pagerank_gpu();
	g.printGraphInfo();

}