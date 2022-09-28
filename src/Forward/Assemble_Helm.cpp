#include "../General/includefile.h"
#include "Assemble_Helm.h"

bool Assemble_Helm(int nz, int nx, Eigen::SparseMatrix<std::complex<float>>& H, Eigen::MatrixXcf& MADI){
    Eigen::MatrixXcf A5 = MADI.row(4).transpose();
    int D5 = 0;
    Eigen::MatrixXcf A2 = MADI.row(1).transpose();
    Eigen::MatrixXcf A4 = MADI.row(3).transpose();
    Eigen::MatrixXcf A6 = MADI.row(5).transpose();
    Eigen::MatrixXcf A8 = MADI.row(7).transpose();
    int D2 = -2 * nz; int D4 = -2; int D6 = 2; int D8 = 2 * nz;
    Eigen::MatrixXcf A1 = MADI.row(0).transpose();
    Eigen::MatrixXcf A3 = MADI.row(2).transpose();
    Eigen::MatrixXcf A7 = MADI.row(6).transpose();
    Eigen::MatrixXcf A9 = MADI.row(8).transpose();

    Eigen::MatrixXcf A1p(A1.rows(), A1.cols()), A1m(A1.rows(), A1.cols());
    A1p.setZero(); A1m.setZero();
    Eigen::MatrixXcf A3p(A3.rows(), A3.cols()), A3m(A3.rows(), A3.cols());
    A3p.setZero(); A3m.setZero();
    Eigen::MatrixXcf A7p(A7.rows(), A7.cols()), A7m(A7.rows(), A7.cols());
    A7p.setZero(); A7m.setZero();
    Eigen::MatrixXcf A9p(A9.rows(), A9.cols()), A9m(A9.rows(), A9.cols());
    A9p.setZero(); A9m.setZero();
    
    //Eigen::initParallel();
    //int nthreads = Eigen::nbThreads();
    //#pragma omp parallel firstprivate(A1, A3, A7, A9)
    //#pragma omp for

    for (int i = 0; i < MADI.cols() / 2; i++){
        A1p.row(2 * i) = A1.row(2 * i);
        A1m.row(2 * i + 1) = A1.row(2 * i + 1);
        A3p.row(2 * i) = A3.row(2 * i);
        A3m.row(2 * i + 1) = A3.row(2 * i + 1);
        A7p.row(2 * i) = A7.row(2 * i);
        A7m.row(2 * i + 1) = A7.row(2 * i + 1);
        A9p.row(2 * i) = A9.row(2 * i);
        A9m.row(2 * i + 1) = A9.row(2 * i + 1);
    }
    int D1p = D2 - 1; int D1m = D2 - 3; int D3p = D2 + 3; int D3m = D2 + 1;
    int D7p = D8 - 1; int D7m = D8 - 3; int D9p = D8 + 3; int D9m = D8 + 1;

    Eigen::MatrixXi d(1, 13);
    d << D1m, D1p, D2, D3m, D3p, D4, D5, D6, D7m, D7p, D8, D9m, D9p;
    Eigen::MatrixXcf A(MADI.cols(), d.cols()); A.setZero();
    A.block(0, 0, A1m.rows() - (-D1m), 1) = A1m.block(-D1m, 0, A1m.rows() - (-D1m), 1);
    A.block(0, 1, A1p.rows() - (-D1p), 1) = A1p.block(-D1p, 0, A1p.rows() - (-D1p), 1);
    A.block(0, 2, A2.rows() - (-D2), 1) = A2.block(-D2, 0, A2.rows() - (-D2), 1);
    A.block(0, 3, A3m.rows() - (-D3m), 1) = A3m.block(-D3m, 0, A3m.rows() - (-D3m), 1);
    A.block(0, 4, A3p.rows() - (-D3p), 1) = A3p.block(-D3p, 0, A3p.rows() - (-D3p), 1);
    A.block(0, 5, A4.rows() - (-D4), 1) = A4.block(-D4, 0, A4.rows() - (-D4), 1);
    A.col(6) = A5;
    A.block(D6, 7, A6.rows() - D6, 1) = A6.block(0, 0, A6.rows() - D6, 1);
    A.block(D7m, 8, A7m.rows() - D7m, 1) = A7m.block(0, 0, A7m.rows() - D7m, 1);
    A.block(D7p, 9, A7p.rows() - D7p, 1) = A7p.block(0, 0, A7p.rows() - D7p, 1);
    A.block(D8, 10, A8.rows() - D8, 1) = A8.block(0, 0, A8.rows() - D8, 1);    
    A.block(D9m, 11, A9m.rows() - D9m, 1) = A9m.block(0, 0, A9m.rows() - D9m, 1);
    A.block(D9p, 12, A9p.rows() - D9p, 1) = A9p.block(0, 0, A9p.rows() - D9p, 1);

    //Eigen::SparseMatrix<std::complex<float>> H(2 * nx * nz, 2 * nx * nz);
    //H.reserve(A.nonZeros());
    spdiags<std::complex<float>>(H, A, d, 2 * nx * nz, 2 * nx * nz);

    A1.resize(0, 0); A2.resize(0, 0); A3.resize(0, 0); A4.resize(0, 0);
    A5.resize(0, 0); A6.resize(0, 0); A7.resize(0, 0); A8.resize(0, 0); A9.resize(0, 0);
    A1p.resize(0, 0); A1m.resize(0, 0); A3p.resize(0, 0); A3m.resize(0, 0);
    A7p.resize(0, 0); A7m.resize(0, 0); A9p.resize(0, 0); A9m.resize(0, 0);
    d.resize(0, 0);
    A.resize(0, 0);

    return 1;
}

