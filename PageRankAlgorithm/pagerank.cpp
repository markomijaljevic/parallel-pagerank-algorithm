#include <iostream>
#include <vector>
#include <string>
#define d 0.85
#define N 5
using namespace std;

typedef vector<vector<double>> Matrix;
typedef vector<double> MatrixRow;

struct Page {
	size_t index;
	string link;
	double PR=0;
	vector<Page> outgoingLinks;
	vector<Page> setOfPagesLinkingToPage;
};

void printVectorMatrix(const Matrix& matrix) {

	for (const MatrixRow &r : matrix)
	{
		for (double x : r) 
			std::cout << x << ' ';
		std::cout << std::endl;
	}
}

MatrixRow sumMatrixRows(const Matrix& matrix) {
	
	MatrixRow sumRow(N);
	double sum = 0;
	for (const MatrixRow &r : matrix) {
		for (double x : r) {
			sum += x;
		}
		sumRow.push_back(sum);
		sum = 0;
	}
	return sumRow;
}

MatrixRow sumMatrixColumns(const Matrix& matrix) {
	
	MatrixRow sumCol(N);
	int i = 0;
	for (const MatrixRow &r : matrix) {
		for (double x : r) {
			sumCol[i] += x;
			i++;
		}
		i = 0;
	}
	return sumCol;
}

void pageRank(const vector<Page>& webPages) {

	Matrix PageRank;
	MatrixRow PageData(N);

	for (const Page& p : webPages) {
		if (p.setOfPagesLinkingToPage.empty()) {
			fill(PageData.begin(), PageData.end(), 0);
			PageRank.push_back(PageData);
			//continue;
		}
		else {
			for (vector<Page>::const_iterator it = p.setOfPagesLinkingToPage.begin() ; it != p.setOfPagesLinkingToPage.end(); it++) {
					PageData[p.setOfPagesLinkingToPage[i].index] = p.setOfPagesLinkingToPage[i].outgoingLinks.size();
					//cout << p.setOfPagesLinkingToPage[i].outgoingLinks.size() << endl;
				//cout << it->outgoingLinks.size() << " ";
			}
			cout << endl;
			PageRank.push_back(PageData);
			fill(PageData.begin(), PageData.end(), 0);
		}
	}

	printVectorMatrix(PageRank);


}

int main() {

	Page p1,p2,p3,p4,p5;
	vector<Page> webPages;
	p1.link = "www.p1.com";
	p2.link = "www.p2.com";
	p3.link = "www.p3.com";
	p4.link = "www.p4.com";
	p5.link = "www.p5.com";

	p1.index = 0;
	p2.index = 1;
	p3.index = 2;
	p4.index = 3;
	p5.index = 4;

	p1.outgoingLinks.push_back(p2);
	p1.outgoingLinks.push_back(p3);

	p2.outgoingLinks.push_back(p3);
	p2.outgoingLinks.push_back(p4);
	p2.setOfPagesLinkingToPage.push_back(p1);

	p3.outgoingLinks.push_back(p5);
	p3.setOfPagesLinkingToPage.push_back(p1);
	p3.setOfPagesLinkingToPage.push_back(p2);
	p3.setOfPagesLinkingToPage.push_back(p4);
	p3.setOfPagesLinkingToPage.push_back(p5);

	p4.outgoingLinks.push_back(p3);
	p4.setOfPagesLinkingToPage.push_back(p2);

	p5.outgoingLinks.push_back(p3);
	p5.setOfPagesLinkingToPage.push_back(p3);

	webPages.push_back(p1);
	webPages.push_back(p2);
	webPages.push_back(p3);
	webPages.push_back(p4);
	webPages.push_back(p5);

	pageRank(webPages);
}
