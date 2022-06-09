#include <iostream>
#include <cmath>

#include "matrix.h"
using namespace std;

//---------------------------------
//     通常のコンストラクタ
//---------------------------------
Matrix::Matrix(int i, int j)
{
  //  i,j のチェック
  if( i<1 || j<1 ){
    cerr << "err Matrix::Matrix" <<endl;
    exit(1);
  }

  row = i;
  column = j;

  //  配列のメモリ領域を動的に確保
  p_top = new double*[row+1];
  *p_top = new double[row*column+1]; 
  // +1 ga daiji kore tukenaito bagu ga deta
  for(int k=1; k<=row; k++)
    *(p_top+k) = *p_top+((k-1)*column);

  //  値の初期化
  for(int k1=1; k1<=row; k1++){
    for(int k2=1; k2<=column; k2++){
      p_top[k1][k2] = 0;
    }
  }
}

//---------------------------------
//     コピーコンストラクタ
//---------------------------------
Matrix::Matrix(const Matrix &cp)
{
  row = cp.row;
  column = cp.column;

  //  配列のメモリ領域を動的に確保
  p_top = new double*[row+1];
  *p_top = new double[row*column+1]; 
  // +1 ga daiji kore tukenaito bagu ga deta
  for(int k=1; k<=row; k++)
    *(p_top+k) = *p_top+((k-1)*column);

  //  値のコピー
  for(int k1=1; k1<=row; k1++){
    for(int k2=1; k2<=column; k2++){
      p_top[k1][k2] = cp.p_top[k1][k2];
    }
  }
}

//----------------------
//   デストラクタ
//----------------------
Matrix::~Matrix()
{
  delete [] *p_top; 
  delete [] p_top;
}

//------------------------------
//   行列の大きさを変える  値は０
//------------------------------
void Matrix::change_size(int i, int j)
{
  //  i,j のチェック
  if( i<1 || j<1 ){
    cerr << "err Matrix::change_size" <<endl;
    exit(1);
  }

  delete [] *p_top; 
  delete [] p_top;

  row = i;
  column = j;

  //  配列のメモリ領域を動的に確保
  p_top = new double*[row+1];
  *p_top = new double[row*column+1]; 
  // +1 ga daiji kore tukenaito bagu ga deta
  for(int k=1; k<=row; k++)
    *(p_top+k) = *p_top+((k-1)*column);

  //  値の初期化
  for(int k1=1; k1<=row; k1++){
    for(int k2=1; k2<=column; k2++){
      p_top[k1][k2] = 0.0;
    }
  }

}

//------------------------------------
//     代入
//------------------------------------
Matrix Matrix::operator=(const Matrix &a)
{
  if( row != a.row || column != a.column ){
    change_size(a.row, a.column);
  }

  for(int i=1; i<=row; i++){
    for(int j=1; j<=column; j++){
      p_top[i][j] = a.p_top[i][j];
    }
  }
  return(*this);
}

//------------------------------------
//       行列の加算
//------------------------------------
Matrix Matrix::operator+(const Matrix &a)
{
  if( row != a.row || column != a.column ){
    cerr << "err Matrix::operator+" <<endl;
    cerr << "  not equal matrix size" <<endl;
    exit(0);
  }

  Matrix r(row, column);
  for(int i=1; i<=row; i++){
    for(int j=1; j<=column; j++){
      r.p_top[i][j] = p_top[i][j] + a.p_top[i][j];
    }
  }
  return(r);
}

//------------------------------------
//       行列の減算
//------------------------------------
Matrix Matrix::operator-(const Matrix &a)
{
  if( row != a.row || column != a.column ){
    cerr << "err Matrix::operator-" <<endl;
    cerr << "  not equal matrix size" <<endl;
    exit(0);
  }

  Matrix r(row, column);
  for(int i=1; i<=row; i++){
    for(int j=1; j<=column; j++){
      r.p_top[i][j] = p_top[i][j] - a.p_top[i][j];
    }
  }
  return(r);
}

//------------------------------------
//       行列の積
//------------------------------------
Matrix Matrix::operator*(const Matrix &a)
{
  if( column != a.row ){
    cerr << "err Matrix::operator*" <<endl;
    cerr << "  not equal matrix size" <<endl;
    exit(0);
  }

  Matrix r(row, a.column);
  for(int i=1; i<=row; i++){
    for(int j=1; j<=a.column; j++){
      for(int k=1; k<=column; k++){
	r.p_top[i][j] += p_top[i][k] * a.p_top[k][j];
      }
    }
  }
  return(r);
}

//------------------------------------
//       行列のHadamard product
//------------------------------------
Matrix Matrix::operator&(const Matrix &a)
{
  if( row != a.row || column != a.column ){
    cerr << "err Matrix::operator&" <<endl;
    cerr << "  not equal matrix size" <<endl;
    exit(0);
  }

  Matrix r(row, column);
  for(int i=1; i<=row; i++){
    for(int j=1; j<=column; j++){
      r.p_top[i][j] = p_top[i][j] * a.p_top[i][j];
    }
  }
  return(r);
}

//--------------------------------------
//       行列の定数倍
//--------------------------------------
Matrix operator*(const Matrix &a, double b)
{
  Matrix r(a.row, a.column);
  for(int i=1; i<=a.row; i++){
    for(int j=1; j<=a.column; j++){
      r[i][j] = b * a.p_top[i][j];
    }
  }
  return(r);
}
Matrix operator*(double b, const Matrix &a)
{
  Matrix r(a.row, a.column);
  for(int i=1; i<=a.row; i++){
    for(int j=1; j<=a.column; j++){
      r[i][j] = b * a.p_top[i][j];
    }
  }
  return(r);
}

//----------------------------------------
//  単位行列にする
//----------------------------------------
void Matrix::unit_matrix()
{
  if(row != column){
    cerr <<"err in Matrix::unit_matrix()" <<endl;
    exit(0);
  }

  int n = row;
  double** a = p_top;
  for(int i=1; i<=n; i++){
    for(int j=1; j<=n; j++){
      a[i][j] = 0;
      if(i == j) a[i][j] = 1;
    }
  }

}

//----------------------------------------
//  転置行列をかえす
//----------------------------------------
Matrix Matrix::transposed()
{
  Matrix t(column, row);
  double** a = p_top;

  for(int i=1; i<=row; i++){
    for(int j=1; j<=column; j++){
      t[j][i] = a[i][j];
    }
  }
  return(t);
}
