#pragma once
#include <cassert>
#include <string>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#define EIGEN_DENSEBASE_PLUGIN "EigenDenseBaseAddons.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "adept.h"

std::string to_lower_case(std::string s);
std::vector<std::vector<std::string> > cross(std::vector<std::vector<std::string> > vectors);
unsigned int popCount(unsigned int i);

typedef Eigen::Matrix<adept::adouble, Eigen::Dynamic, 1> AColumnVector;
typedef Eigen::Matrix<adept::adouble, 1, Eigen::Dynamic> ARowVector;
typedef Eigen::Matrix<adept::adouble, Eigen::Dynamic, Eigen::Dynamic> AMatrix;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVector;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

ColumnVector convert(const AColumnVector& v);
RowVector convert(const ARowVector& v);
Matrix convert(const AMatrix& m);

AMatrix cwiseSqrt(const AMatrix& m);
AMatrix cwiseSigmoid(const AMatrix& m);
AMatrix cwiseTanh(const AMatrix& m);
void VerifySanity(const Matrix& vector);
void VerifySanity(const AMatrix& vector);

double log_sum_exp(std::vector<double> v);
adept::adouble log_sum_exp(std::vector<adept::adouble> v);

AMatrix ReadMatrix(std::string filename);
void WriteMatrix(const AMatrix& matrix, std::string filename);

AMatrix amult(const AMatrix& a, const Matrix& b);
AMatrix amult(const Matrix& a, const AMatrix& b);

/*namespace boost {
  namespace serialization {
    template<class Archive>
    inline void save(Archive& ar, const adept::adouble& a, const unsigned int file_version) {
      double val = a.value();
      ar & val;
    }

    template<class Archive>
    inline void load(Archive& ar, adept::adouble& a, const unsigned int file_version) {
      double val;
      ar & val;
      a = val;
    }

    template<class Archive>
    inline void serialize(Archive& ar, adept::adouble& a, const unsigned int file_version) {
      split_free(ar, a, file_version);
    }
  }
}*/

/*template <class T>
inline void hash_combine(size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}*/
