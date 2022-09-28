#include "../General/includefile.h"
#include "par_FDFD_anelastic.h"
#include "Make_Helm_Anelastic.h"
#include "Assemble_Helm.h"

bool par_FDFD_anelastic(Eigen::MatrixXf& model, Eigen::RowVectorXf& frequency, \
                        float omega0, Eigen::SparseMatrix<float>& S, Eigen::MatrixXcf& fwave, \
                        std::vector<Eigen::SparseMatrix<std::complex<float>>>& U, \
                        int PML_thick, int nz, int nx, int dz){
    int nzPML = nz + 2 * PML_thick; int nxPML = nx + 2 * PML_thick;
    int lf = frequency.size();
        Eigen::SparseMatrix<std::complex<float>> A(2 * nxPML * nzPML, 2 * nxPML * nzPML);
        Eigen::MatrixXcf MADI(9, nzPML * nxPML * 2);
        MADI.setZero();
        Eigen::SparseMatrix<std::complex<float>> temp(nzPML * nxPML * 2, fwave.cols());
        Eigen::SparseMatrix<std::complex<float>> fcoeff(fwave.cols(), fwave.cols());
        Eigen::SparseMatrix<std::complex<double>> u(nzPML * nxPML * 2, fwave.cols());
Eigen::initParallel();
int nthreads = Eigen::nbThreads();
#pragma omp parallel firstprivate(MADI, A, temp, fcoeff, u)
#pragma omp for
        // How about other variables? Are they shared by multithreads?
    for (int n = 0; n < lf; n++){
        //std::clock_t start, end;
        //start = std::clock();
        float omega = 2 * pi * frequency(n);

        bool flag_MADI = Make_Helm_Anelastic(MADI, model, nz, nx, omega, omega0, dz, PML_thick);
        bool flagA = Assemble_Helm(nzPML, nxPML, A, MADI);
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver(A.cast<std::complex<double>>());

        spdiags<std::complex<float>>(fcoeff, fwave.row(n), Eigen::MatrixXi::Zero(1, 1), fwave.cols(), fwave.cols());
        temp = S * fcoeff;
        u = solver.solve(temp.cast<std::complex<double>>());
        U[n] = u.cast<std::complex<float>>();

        A.setZero();
        MADI.setZero();
        temp.setZero();
        fcoeff.setZero();
        u.setZero();
        //end = std::clock();
        //std::cout.precision(5);
        //std::cout << "=== Finished " << n + 1<< "th freq.===" << std::endl;
        ////std::cout << "The time for " << n + 1 << "th forward modeling is: " << (float)(end - start) / CLOCKS_PER_SEC << "s." << std::endl;
        //std::cout << "The time for " << n + 1 << "th forward modeling is: " << (float)(end - start) / CLOCKS_PER_SEC / omp_get_num_threads() << "s." << std::endl;
    }
    A.resize(0, 0); A.data().squeeze();
    MADI.resize(0, 0);
    temp.resize(0, 0); temp.data().squeeze();
    fcoeff.resize(0, 0); fcoeff.data().squeeze();
    u.resize(0, 0); u.data().squeeze();
    std::cout << "U done." << std::endl;
    return 1;

}


