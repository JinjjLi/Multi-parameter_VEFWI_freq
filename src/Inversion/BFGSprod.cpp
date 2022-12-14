#include "../General/includefile.h"
#include "BFGSprod.h"

Eigen::MatrixXf BFGSprod(Eigen::MatrixXf& v, Eigen::MatrixXf& gstore, Eigen::MatrixXf& mstore, Eigen::SparseMatrix<float>& H0){
    Eigen::MatrixXf q = v;
    Eigen::MatrixXf y = Eigen::MatrixXf(gstore.rows(), 1);
    Eigen::MatrixXf s = Eigen::MatrixXf(gstore.rows(), 1);
    Eigen::MatrixXf rho = Eigen::MatrixXf(gstore.cols() - 1, 1);
    Eigen::MatrixXf alpha = Eigen::MatrixXf(gstore.cols() - 1, 1);
    y.setZero(); s.setZero();
    rho.setZero(); alpha.setZero();
    int i = gstore.cols() - 2;
    while(i >= 0){
        y = gstore.col(i + 1) - gstore.col(i);
        s = mstore.col(i + 1) - mstore.col(i);
        rho(i) = (y.transpose() * s).value();
        rho(i) = 1.0 / rho(i);
        alpha(i) = (s.transpose() * q).value();
        alpha(i) *= rho(i);
        q -= alpha(i) * y;
        i--;
    }
    Eigen::MatrixXf r = H0 * q;
    for (int i = 0; i < gstore.cols() - 1; i++){
        y = gstore.col(i + 1) - gstore.col(i);
        s = mstore.col(i + 1) - mstore.col(i);
        float beta = (rho(i) * y.transpose() * r).value();
        r += s * (alpha(i) - beta);
    }
    q.resize(0, 0); y.resize(0, 0); s.resize(0, 0);
    rho.resize(0, 0); alpha.resize(0, 0);
    return r;
}

