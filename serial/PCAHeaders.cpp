#include <bits/stdc++.h>
#include <string>
using namespace std;


//===========================================================================================================//

vector<std::vector<double> > extractData(string s){

	std::vector< std::vector<double> > A;
	fstream fs;
	fs.open(s.c_str(),ios::in|ios::out);
	string line;

	while(getline(fs,line))
	{
		stringstream ss(line);
		double a;
		std::vector<double> temp;
		while(ss>>a)
		{
			temp.push_back(a);

		}

		A.push_back(temp);
	}

	return A;
}


//=========================calculating mean=========================================================================//



vector<std::vector<double> > calculateMean(vector<std::vector<double> > A, int rows, int cols){
	double temp =0;

	vector<std::vector<double> >mean;

	std::vector<double>t;

	for (int i = 0; i < cols; ++i)
	{

		for (int j = 0; j < rows ; ++j)
		{
			temp+=A[j][i];
		}

		t.push_back(double(temp)/cols);
		mean.push_back(t);
		t.clear();
		temp = 0;
	}


	return mean;

}



//============for testing use this for loops==============================================================================//

void testing(std::vector<std::vector<double> > A,int rows, int cols)
{
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			cout<<A[i][j]<<" ";
		}		
		cout<<endl;

	}

}

//============function for Matrix multiplication===========================================================================//


vector<std::vector<double> > matMul(vector<std::vector<double> >A,vector<std::vector<double> >B,int rows1,int rows2,int cols2)
{

std::vector<std::vector<double> > v;

	for (int i = 0; i < rows1; ++i)
	{
		std::vector<double> t;
		for (int j = 0; j < rows2 ; ++j)
		{
			double temp =0;
			for (int k = 0; k < cols2; ++k)
			{
				temp+=A[i][k]*B[k][j];

			}
			t.push_back(temp);
		}
		v.push_back(t);
	}

	return v;

}


//=================module to subtract matrix from mean========================================================================//


vector<std::vector<double> > subtractMean(vector< std::vector<double> > A, vector< std::vector<double> > B,int rows1,int cols1)
{
	
	std::vector<std::vector<double> > C(rows1,std::vector<double>(cols1,0.0));

	for(int i = 0 ; i < cols1 ; i++)
	{
		for (int j = 0; j < rows1; ++j)
		{
			C[j][i] = A[j][i]-B[i][0];
		}

	}

	return C;
}

//==============================================================================================//