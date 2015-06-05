#include <algorithm>
#include <cassert>
#include <fstream>
#include <Eigen/Dense>
#include "utils.h"
using namespace std;
using namespace adept;

string to_lower_case(string s) {
  string t = s;
  transform(t.begin(), t.end(), t.begin(), ::tolower);
  return t;
}

vector<vector<string> > cross(vector<vector<string> > vectors ) {
  vector<vector<string> > cross_product;

  // There's exactly one permutation of nothing
  if (vectors.size() == 0) {
    vector<string> empty;
    cross_product.push_back(empty);
    return cross_product;
  }

  vector<unsigned> indices(vectors.size(), 0);
  while (true) {
    vector<string> item;
    item.reserve(vectors.size());
    for (unsigned i = 0; i < vectors.size(); ++i) {
      item.push_back(vectors[i][indices[i]]);
    }
    cross_product.push_back(item);

    for (unsigned i = vectors.size(); i > 0;) {
      --i;
      indices[i]++;
      if (indices[i] == vectors[i].size()) {
        if (i != 0) {
          indices[i] = 0;
        }
        else {
          return cross_product;
        }
      }
      else {
        break;
      }
    }
  }
}

unsigned int popCount(unsigned int i) {
  unsigned int r = 0;
  unsigned int one = 1;
  for (unsigned b = 0; b < 8 * sizeof(unsigned int); ++b) {
    unsigned int mask = one << b;
    if (i & mask) {
      ++r;
    }
  }
  return r;
}

void VerifySanity(const Matrix& vector) {
  for (int i = 0; i < vector.rows(); ++i) {
    for (int j = 0; j < vector.cols(); ++j) {
      // assert vector[i][j] is not inifinity or NaN
    }
  }
}

void VerifySanity(const AMatrix& vector) {
  for (int i = 0; i < vector.rows(); ++i) {
    for (int j = 0; j < vector.cols(); ++j) {
      // assert vector[i][j] is not inifinity or NaN
    }
  }
}

double log_sum_exp(vector<double> v) {
  if (v.size() == 0) {
    return -numeric_limits<double>::infinity();
  }
  double m = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (v[i] > m) {
      m = v[i];
    }
  }

  double sum = 0.0;
  for (unsigned i = 0; i < v.size(); ++i) {
    sum += exp(v[i] - m);
  }
  return m + log(sum);
}

adouble log_sum_exp(vector<adouble> v) {
  if (v.size() == 0) {
    return -numeric_limits<double>::infinity();
  }
  adouble m = v[0];
  for (unsigned i = 1; i < v.size(); ++i) {
    if (v[i] > m) {
      m = v[i];
    }
  }

  adouble sum = 0.0;
  for (unsigned i = 0; i < v.size(); ++i) {
    sum += exp(v[i] - m);
  }
  return m + log(sum);
}


ColumnVector convert(const AColumnVector& v) {
  ColumnVector r(v.size());
  for (int i = 0; i < v.size(); ++i) {
    r(i) = value(v(i));
  }
  return r;
}

RowVector convert(const ARowVector& v) {
  RowVector r(v.size());
  for (int i = 0; i < v.size(); ++i) {
    r(i) = value(v(i));
  }
  return r;
}

Matrix convert(const AMatrix& m) {
  Matrix r(m.rows(), m.cols());
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      r(i, j) = value(m(i, j));
    }
  }
  return r;
}

AMatrix cwiseSqrt(const AMatrix& m) {
  AMatrix r(m.rows(), m.cols());
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      r(i, j) = sqrt(m(i, j));
    }
  }
  return r;
}

AMatrix cwiseSigmoid(const AMatrix& m) {
  AMatrix r(m.rows(), m.cols());
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      r(i, j) = 1.0 / (1 + exp(-m(i, j)));
    }
  }
  return r;
}

AMatrix cwiseTanh(const AMatrix& m) {
  AMatrix r(m.rows(), m.cols());
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      r(i, j) = tanh(m(i, j));
    }
  }
  return r;
}

AMatrix ReadMatrix(string filename) {
  ifstream f(filename);
  vector<vector<double>> mat;
  string line;
  while (f) {
    getline(f, line);
    double d;
    istringstream ss(line);
    vector<double> row;
    while (ss >> d) {
      row.push_back(d);
    }
    mat.push_back(row);
  }

  AMatrix ret(mat.size() - 1, mat[0].size());
  for (unsigned i = 0; i < mat.size() - 1; ++i) {
    for (unsigned j = 0; j < mat[i].size(); ++j) {
      ret(i, j) = mat[i][j];
    }
  }
  return ret;
}

void WriteMatrix(const AMatrix& matrix, string filename) {
  ofstream f(filename);
  for (unsigned i = 0; i < matrix.rows(); ++i) {
    for (unsigned j = 0; j < matrix.cols(); ++ j) {
      f << matrix(i, j);
      if (j != matrix.cols() - 1) {
        f << " ";
      }
    }
    f << "\n";
  } 
  f.close();
}

AMatrix amult(const AMatrix& a, const Matrix& b) {
  assert (a.cols() == b.rows());
  AMatrix r(a.rows(), b.cols());

  for (unsigned int i = 0; i < a.rows(); ++i) {
    for (unsigned int j = 0; j < b.cols(); ++j) {
      adouble s = 0.0;
      for (unsigned int k = 0; k < a.cols(); ++k) {
        s += a(i, k) * b(k, j);
      }
      r(i, j) = s;
    }
  }

  return r;
}

AMatrix amult(const Matrix& a, const AMatrix& b) {
  assert (a.cols() == b.rows());
  AMatrix r(a.rows(), b.cols());

  for (unsigned int i = 0; i < a.rows(); ++i) {
    for (unsigned int j = 0; j < b.cols(); ++j) {
      adouble s = 0.0;
      for (unsigned int k = 0; k < a.cols(); ++k) {
        s += a(i, k) * b(k, j);
      }
      r(i, j) = s;
    }
  }

  return r;
}
