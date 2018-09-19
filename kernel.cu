#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <vector>
#include <math.h>
#include <thrust\reduce.h>
#include <thrust\device_ptr.h>
#include <thrust\device_vector.h>

#define N 9663
#define d 0.85

int global_index = 0; 

struct Node {
	std::string url;
	double PRInLastIter = (double)1 / N;
	double PRInCurrentIter = (double)1 / N;
	int inLinks[N];
	int outLinks[N];
	int index = 0;
	int outLinksIndex = -1;
	int inLinksIndex = -1;
	int iter;

	Node() {}
	Node(std::string);
	~Node(){}

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
	void printTopPageRanks();
};

Node::Node(std::string url) {
	this->url = url;
	this->index = global_index;
	global_index++;
	//std::cout << global_index << std::endl;
}

void Node::isLinkingTo(Node& node) {
	this->outLinksIndex++;
	this->outLinks[this->outLinksIndex]= node.index;
	node.pushInLink(this->index);
}

void Node::pushInLink(int indexOfNode) {
	this->inLinksIndex++;
	this->inLinks[this->inLinksIndex] = indexOfNode;
}

void Graph::addNode(Node node) {
	this->nodes.push_back(node);
}

void Graph::printGraphInfo() {

	long double sumOfPR = 0;
	std::ofstream file;
	file.open("pagerank_results.txt");
	for (Node node : this->nodes) {
		file << "Url: " << node.url << std::endl;
		file << '\t' << "PR: " << (node.PRInCurrentIter / this->normalize_sum)<< std::endl;
		file << '\t' << "In Going Links: " << node.inLinksIndex + 1 << std::endl;
		file << '\t' << "Out Going Links: " << node.outLinksIndex + 1 << std::endl;
		file << std::endl;
		sumOfPR += node.PRInCurrentIter / this->normalize_sum;
	}
	file.close();
	std::cout << "Normalized sum = " << sumOfPR << std::endl;
	std::cout << "Number of iterations = " << this->iterCounter << std::endl;

	printTopPageRanks();
}

void Graph::printTopPageRanks() {

	std::ofstream file;
	file.open("topPageRanks.txt");
	int i = 0;
	Node top_node = this->nodes[0];
	Node temp_node;

	while (true) {
		temp_node = this->nodes[0];
		for (Node n : this->nodes) {
			if (i < 1 && n.PRInCurrentIter >= top_node.PRInCurrentIter) {
				top_node = n;
			}
			if (i >= 1 && n.PRInCurrentIter > temp_node.PRInCurrentIter && n.PRInCurrentIter < top_node.PRInCurrentIter) {
				temp_node = n;
			}
		}

		if (top_node.PRInCurrentIter == temp_node.PRInCurrentIter) {
			break;
		}
		else if(i >= 1)
			top_node = temp_node;
		
		file << "Url: " << top_node.url << std::endl;
		file << '\t' << "PR: " << (top_node.PRInCurrentIter / this->normalize_sum) << std::endl;
		file << '\t' << "In Going Links: " << top_node.inLinksIndex + 1 << std::endl;
		file << '\t' << "Out Going Links: " << top_node.outLinksIndex + 1 << std::endl;
		file << std::endl;

		i++;
	}
	file.close();
}

__global__ void calculatePageRank(Node* nodes,double* normSum, int* iterCounter) {

	int index = threadIdx.x;
	double PR = 0;
	int j;
	*iterCounter = -1;
	
	while (true){

		*iterCounter +=1;

		for (int i = 0; i < nodes[index].inLinksIndex + 1; i++) {
			j = nodes[index].inLinks[i];
			PR += ( nodes[j].PRInCurrentIter / ( nodes[j].outLinksIndex + 1 ));
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
	int* devIterCounter;
	double* devNormSum;
	cudaError_t cudaStatus;

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

	dim3 threadsPerBlock(1024, 1, 1);
	dim3 blocksPerGrid(ceil((double)N / threadsPerBlock.x), 1, 1);

	std::cout <<"Blocks per grid = " << blocksPerGrid.x << std::endl;
	std::cout << "Threads per block = " << threadsPerBlock.x << std::endl;

	calculatePageRank <<< blocksPerGrid, threadsPerBlock>>> (devNodes, devNormSum, devIterCounter);

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

	this->normalize_sum = thrust::reduce(norm_sum.begin(), norm_sum.end(), (double)0, thrust::plus<double>());

	cudaFree(devNodes);
	cudaFree(devNormSum);
	cudaFree(devIterCounter);
}

int main(void) {

	Graph graph;
	std::ifstream dataset("dataset.txt");
	std::string line;
	std::size_t npos;
	std::clock_t start;
	double duration;

	while (std::getline(dataset, line)) {
		
		if (line[0] == 'n') {
			npos = line.find_first_of('h', 4);
			Node node(line.substr(npos, line.length()));
			graph.addNode(node);
		}
		else { // if line[0] == 'e'
			npos = line.find_first_of(' ', 2);
			graph.nodes[std::stoi(line.substr(2, npos - 2))].isLinkingTo(graph.nodes[std::stoi(line.substr(npos + 1, line.length() - npos))]);
		}
	}

	start = std::clock();
	///
	graph.pagerank_gpu();
	///
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "PageRank calculation is finished with time of >>  " << duration <<" sec <<" << std::endl;
	graph.printGraphInfo();
}