#include <iostream>                                                                                                                                                          
#include <tuple>
#include <math.h>
#include <complex>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <vector>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <omp.h>

const float pi = 3.1415926;

template<typename Func>
struct lambda_as_visitor_wrapper : Func {
    lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
    template<typename S,typename I>
    void init(const S& v, I i, I j) { return Func::operator()(v,i,j); }
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
    lambda_as_visitor_wrapper<Func> visitor(f);
    m.visit(visitor);
}

std::vector<int> intersect(std::vector<int> &v1, std::vector<int> &v2);

template <typename Type>
void spdiags(Eigen::SparseMatrix<Type>& A, const Eigen::MatrixXcf& B, const Eigen::MatrixXi& d, int m, int n){
    //Eigen::SparseMatrix<std::complex<float>> A(m, n);
    //Eigen::SparseMatrix<Type> A(m, n);
    A.reserve(B.size());
    typedef Eigen::Triplet<std::complex<float>> T;
    std::vector<T> triplets;
    //triplets.reserve(std::min(m, n) * d.size());

    for (int k = 0; k < d.size(); k++){
        if (d(k) >= 0){
            int i_min = std::max(0, -d(k));
            int i_max = std::min(m - 1, n - d(k) - 1);
            int B_idx_start = d(k);
            //int B_idx_start = m >= n ? d(k) : 0;
            for (int i = i_min; i <= i_max; i++){
                triplets.push_back(T(i, i + d(k), B(B_idx_start + i, k)));
            }
        }
        if (d(k) < 0){
            int i_min = std::max(0, -d(k));
            int i_max = std::min(m - 1, n - d(k) - 1);
            int B_idx_start = 0;
            //int B_idx_start = m >= n ? 0 : -d(k);
            for (int i = i_min; i <= i_max; i++){
                triplets.push_back(T(i, i + d(k), B(B_idx_start + i + d(k), k)));
            }
        }
    }

  A.setFromTriplets(triplets.begin(), triplets.end());
  triplets.clear(); triplets.shrink_to_fit();
}

template <typename Type>
void spdiags_noncomplex(Eigen::SparseMatrix<Type>& A, const Eigen::MatrixXf& B, const Eigen::MatrixXi& d, int m, int n){
    //Eigen::SparseMatrix<std::complex<float>> A(m, n);
    //Eigen::SparseMatrix<float> A(m, n);                                                                               
    //A.reserve(B.size());
    typedef Eigen::Triplet<float> T;
    std::vector<T> triplets;
    //triplets.reserve(std::min(m, n) * d.size());

    for (int k = 0; k < d.size(); k++){
        if (d(k) >= 0){
            int i_min = std::max(0, -d(k));
            int i_max = std::min(m - 1, n - d(k) - 1);
            int B_idx_start = d(k);
            //int B_idx_start = m >= n ? d(k) : 0;
            for (int i = i_min; i <= i_max; i++){
                triplets.push_back(T(i, i + d(k), B(B_idx_start + i, k)));
            }
        }
        if (d(k) < 0){
            int i_min = std::max(0, -d(k));
            int i_max = std::min(m - 1, n - d(k) - 1);
            int B_idx_start = 0;
            //int B_idx_start = m >= n ? 0 : -d(k);
            for (int i = i_min; i <= i_max; i++){
                triplets.push_back(T(i, i + d(k), B(B_idx_start + i + d(k), k)));
            }
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    triplets.clear(); triplets.shrink_to_fit();
    //return A;
}


template<class ArgType, class RowIndexType, class ColIndexType>                                                                                                                                                                                                                                                                                          
class indexing_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
  const ColIndexType &m_colIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                 RowIndexType::SizeAtCompileTime,
                 ColIndexType::SizeAtCompileTime,
                 ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
                 RowIndexType::MaxSizeAtCompileTime,
                 ColIndexType::MaxSizeAtCompileTime> MatrixType;
 
  indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
  {}
 
  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};


template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
mat_indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices);

