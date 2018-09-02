#include "Graph.h"
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void Graph::addNewNode(Node* newNode) {
	//std::cout << "Adding newNode..." << std::endl;
	//std::cout << "Adresa Èvorova na poèetku " << &*newNode << " Tip " << typeid(newNode).name() << " Velicina " << sizeof(newNode) << " Ime " << newNode->getUrl() << std::endl;
	newNode->sizeOfOutLinks = newNode->getOutGoingLinks().size();
	this->Nodes.push_back(newNode);
	//std::cout << std::endl;
}

size_t Graph::getNodeCount() {
	return this->Nodes.size();
}


Node* Graph::getInLinksNodes(thrust::host_vector<Node*> links) {

	//std::cout << "adresa inlinksa u funkciju " <<  &links[0] << std::endl;

	this->inLinks = new Node[links.size()];
	int i = 0;
	for (Node* node_ptr : links) {
		inLinks[i] = *node_ptr;
		i++;
	}
	//std::cout << "adresa u nizu " << &inLinks[0] << " ime " << inLinks[0].getUrl() << std::endl;
	return &inLinks[0];
}

void Graph::printGraphInfo() {
	for (Node* node : this->Nodes) {
		std::cout << "Url: " << node->getUrl() << std::endl;
		std::cout << "PR: " << (node->getCurrentPR() / this->normalize_sum ) << std::endl;
		std::cout << "In Going Links: " << node->getInGoingLinks().size() << std::endl;
		std::cout << "Out Going Links: " << node->getOutGoingLinks().size() << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Number of iterations: " << this->iter_counter << std::endl;
	std::cout << std::endl;
}


__global__ void calculatePR( Node* dev_inLinks, float* dev_PR) {

	int i = threadIdx.x;
	dev_PR[i] = ( dev_inLinks[i].PRinLastIter / dev_inLinks[i].sizeOfOutLinks );
}

__global__ void updateLastPR( Node* dev_Nodes, bool* flag, float* dev_sum) {

	int i = threadIdx.x;
	if (dev_Nodes[i].PRinLastIter != dev_Nodes[i].PRinCurrentIter)
		*flag = true;

	dev_Nodes[i].PRinLastIter = dev_Nodes[i].PRinCurrentIter;
	dev_sum[i] = dev_Nodes[i].PRinCurrentIter;
}


void Graph::PageRank_GPU() {
	
	int sizeOfInLinks;
	Node* inLinks;
	Node* dev_inLinks;
	Node* dev_Nodes;
	Node* temp_Nodes;
	bool flag = true;
	bool* dev_flag;
	float* dev_normalize_sum;
	float* dev_PR;
	//float* niz_PR;
	float PR;

	cudaError_t cudaStatus;


	while (flag) {

		flag = false;
		this->normalize_sum = 0;

		for (Node* node : this->Nodes) {

			sizeOfInLinks = node->getInGoingLinks().size();

			//std::cout << "Ime covra " << node->getUrl() << " adresa " << &node << std::endl;

			if (sizeOfInLinks > 0) {

				inLinks = getInLinksNodes(node->inlinks);

				//std::cout << "adresa inlinksa " << &node->inlinks << std::endl;

				//for (int i = 0; i < sizeOfInLinks; i++) {
				//	std::cout << "       Ime covra " << inLinks[i].getUrl() << " adresa " << &inLinks[i]<< std::endl;
				//}

				cudaStatus = cudaSetDevice(0);
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
					return;
				}

				cudaStatus = cudaMalloc((void**)&dev_inLinks, sizeOfInLinks * sizeof(Node));
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaMalloc failed" << std::endl;
					return;
				}

				cudaStatus = cudaMalloc((void**)&dev_PR, sizeOfInLinks * sizeof(float));
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaMalloc failed" << std::endl;
					return;
				}

				//niz_PR = new float[sizeOfInLinks];

				cudaStatus = cudaMemcpy(dev_inLinks, inLinks, sizeOfInLinks * sizeof(Node), cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaMemcpy failed" << std::endl;
					return;
				}

				calculatePR << < 1, sizeOfInLinks >> > (dev_inLinks, dev_PR);

				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					std::cout << "calculatePR launch failed: " <<  cudaGetErrorString(cudaStatus) << std::endl;;
					return;
				}

				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << "after launching calculatePR!" << std::endl;
					 return;
				}

				/*cudaStatus = cudaMemcpy(inLinks, dev_inLinks, sizeOfInLinks * sizeof(Node), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					std::cout << "cudaMemcpy failed" << std::endl;
					return;
				}*/
				//cudaMemcpy(niz_PR, dev_PR, sizeOfInLinks * sizeof(float), cudaMemcpyDeviceToHost);

				//wrap raw pointer with a device_ptr
				thrust::device_ptr<float> dev_ptr_PR(dev_PR);
				//copy memory to a new dev vector , and use vector 
				thrust::device_vector<float> PageRankValue(dev_ptr_PR, dev_ptr_PR + sizeOfInLinks);

				//thrust::copy(PageRankValue.begin(), PageRankValue.end(), std::ostream_iterator<float>(std::cout, "\n"));
				PR = thrust::reduce(PageRankValue.begin(), PageRankValue.end(), (float)0, thrust::plus<float>());
				/*PageRankValue.clear();
				PageRankValue.shrink_to_fit();*/
				//std::cout << "Pr nakon reduce " << PR << std::endl;

				delete[] this->inLinks;
				//delete[] niz_PR;
				this->inLinks = 0;
				PageRankValue.clear();
				PageRankValue.shrink_to_fit();
			}

			PR = (1 - d) + (d * PR);

			node->setPR(PR);
			//std::cout << "Gpu -> " << PR << std::endl;
			PR = 0;

			if (sizeOfInLinks > 0) {
				cudaFree(dev_inLinks);
				cudaFree(dev_PR);
				dev_PR = 0;
				dev_inLinks = 0;
				inLinks = 0;
			}

		}//end of for

		temp_Nodes = getInLinksNodes(this->Nodes);
	
		sizeOfInLinks = this->Nodes.size();// kolicina cvorova u mreži

		cudaStatus = cudaMalloc((void**)&dev_Nodes, sizeOfInLinks * sizeof(Node));
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc failed" << std::endl;
			return;
		}

		cudaStatus = cudaMalloc((void**)&dev_flag, sizeof(bool));
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc failed" << std::endl;
			return;
		}

		cudaStatus = cudaMalloc((void**)&dev_normalize_sum, sizeOfInLinks * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc failed" << std::endl;
			return;
		}


		cudaStatus = cudaMemcpy(dev_Nodes, temp_Nodes, sizeOfInLinks * sizeof(Node), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed" << std::endl;
			return;
		}

		updateLastPR << <1, sizeOfInLinks >> > (dev_Nodes, dev_flag, dev_normalize_sum);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			std::cout << "updateLastPR launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;;
			return;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << "after launching updateLastPR!" << std::endl;
			return;
		}

		cudaStatus = cudaMemcpy(temp_Nodes, dev_Nodes, sizeOfInLinks * sizeof(Node), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed" << std::endl;
			return;
		}

		cudaStatus = cudaMemcpy(&flag, dev_flag, sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed" << std::endl;
			return;
		}

		thrust::device_ptr<float> dev_ptr_sum(dev_normalize_sum);
		thrust::device_vector<float> norm_sum(dev_ptr_sum, dev_ptr_sum + sizeOfInLinks);

		this->normalize_sum = thrust::reduce(norm_sum.begin(), norm_sum.end(), (float)0, thrust::plus<float>());
		//std::cout << this->normalize_sum << std::endl;


		cudaFree(dev_Nodes);
		cudaFree(dev_flag);
		cudaFree(dev_normalize_sum);
		delete[] this->inLinks;
		this->inLinks = 0;

		norm_sum.clear();
		norm_sum.shrink_to_fit();

		this->iter_counter++;

	}//end of while


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!" << std::endl;
		return;
	}


}//end of for


