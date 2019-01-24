#include <string>
#include <bits/stdc++.h>
#include <Eigenvalues> 
#include <cstring> 
#include "PCAHeaders.cpp" 
#include <time.h>

using namespace Eigen;
using namespace std;

std::vector< std::vector<double> > A,MCD;





int main(int argc, char const *argv[])
{

//==============taking input from file and feeding it to matrix===============================//

	// if(argc == 1){
	// 	cout<<"\n=========== ENTER DATA FILE NAME ==============\n\n";
	// 	cout<<" <<<<<<  FORMAT  >>>> : ./EXE FILENAME.TXT"<<endl<<endl;
	// 	return 0;
	// }

	// A = extractData(argv[1]);
	int cap = 1024*1024;

	for (int i = 0; i <1024 ; ++i)
	{
		std::vector<double> temp;
		for (int j = 0; j < 1024; ++j)
		{
			/* code */
			temp.push_back(i*1024+j);
		}
		A.push_back(temp);

	}

	int rows = A.size();
	int cols = A[0].size();

//=================Now finding mean of given matrix taking each column as one sample=============//


vector<std::vector<double> >mean;
clock_t meanStart = clock();
cout<<"making data mean centred..."<<endl;
mean = calculateMean(A, rows, cols);


// for (int i = 0; i < mean.size(); ++i)
// {
// 	/* code */
// 	cout<<mean[i][0]<<" ";
// }

//=================mean centering data=============================================================//

MCD = subtractMean(A,mean,A.size(),A[0].size());
printf("time taken for making data mean centred: %.2fs\n", (double)(clock() - meanStart)/CLOCKS_PER_SEC);


//testing(MCD,MCD.size(),MCD[0].size());

//=========computing covariance matrix multiplying rowwise without taking transpose================//


clock_t covStart = clock();
cout<<"making co-variance matrix..."<<endl;

std::vector<std::vector<double> > covMatrix(rows,std::vector<double>(rows,0.0));

for (int i = 0; i < rows; ++i)
{
	for (int j = 0; j < cols ; ++j)
	{
		double temp =0;
		for (int k = 0; k < rows; ++k)
		{
			temp+=MCD[k][i]*MCD[k][j];

		}

		covMatrix[i][j] = double(temp)/(rows-1);

	}
}

printf("time taken for making co-variance matrix: %.2fs\n", (double)(clock() - covStart)/CLOCKS_PER_SEC);

//========putting data to matrix to find eigen values using eigen package===========================//

	int r = covMatrix.size();
	int c = covMatrix[0].size();

	MatrixXd Data(r,c);

	// for (int i = 0; i < r; ++i)
	// {
	// 	for (int j = 0; j < c; ++j)
	// 		{
	// 			Data(i,j) = covMatrix[i][j];
	// 		}	

	// }


//============computing eigen values and eigen vectors=============================================//

	clock_t eigStart = clock();
	cout<<"finding eigen values and vectors of co-variance matrix..."<<endl;
	EigenSolver<MatrixXd> es(Data);
	es.compute(Data,true);

	MatrixXd eigenvalues = es.pseudoEigenvalueMatrix();
	MatrixXd eigenvectors = es.pseudoEigenvectors();

	

	printf("time taken for finding eigen values & eigen vectors: %.2fs\n", (double)(clock() - eigStart)/CLOCKS_PER_SEC);

	std::vector<double> ev,eigvalues;
	std::vector<int> eigPos;
	std::vector<double>::iterator it;


//===============separating eigen values from matrix=========================//
	

	for (int i = 0; i < r; ++i)
	{
		ev.push_back(eigenvalues(i,i));	
		eigvalues.push_back(eigenvalues(i,i));	

	}

	sort(ev.begin(),ev.end(),std::greater<double>());

	//======finding eigen positions===========================//

	for (int i = 0; i < r; ++i)
	{
		it = find(eigvalues.begin(),eigvalues.end(),ev[i]);
		eigPos.push_back(it-eigvalues.begin());
	}


	std::vector<std::vector<double> > pca(r,std::vector<double>(c,0.0)),Y;

	for (int i = 0; i < eigenvectors.cols(); ++i)
	{
		for (int j = 0; j < eigenvectors.rows() ; ++j)
		{
			pca[j][i] = eigenvectors.col(eigPos[i])[j];
		}
	}




	cout<<"\n\n===================TRANSPOSED PCA Running=================\n"<<endl;
	//testing(pca,pca.size(),1);
	clock_t mm2Start = clock();
	cout<<"projecting data..."<<endl;
	Y = matMul(MCD,pca,MCD[0].size(),pca[0].size(),pca.size());
	//cout<<"\n\n====================PROJECTED DATA================\n"<<endl;
	printf("time taken for projecting: %.2fs\n", (double)(clock() - mm2Start)/CLOCKS_PER_SEC);

	//testing(Y,Y.size(),1);

	cout<<endl;
	cout<<"\n\n===================================================\n"<<endl;
	return 0;
}
