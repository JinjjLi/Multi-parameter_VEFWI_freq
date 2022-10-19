#include "../General/includefile.h"
#include "Get_data_anelastic.h"
#include "par_FDFD_anelastic.h"

bool Get_data_anelastic(std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::RowVectorXf& freq, Eigen::MatrixXcf& fwave, Eigen::MatrixXf& MODEL, Eigen::SparseMatrix<float>& R, \
                                                 float omega0, Eigen::SparseMatrix<float>& S, int PML_thick, int nz, int nx, int dz){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    std::vector<Eigen::SparseMatrix<std::complex<float>>> U(freq.size());
    bool flag_par = par_FDFD_anelastic(MODEL, freq, omega0, S, fwave, U, PML_thick, nz, nx, dz);
    //std::vector<Eigen::MatrixXcf> D(freq.size());
Eigen::initParallel();
int nthreads = Eigen::nbThreads();
//#pragma omp parallel firstprivate(R, U)
#pragma omp for
    for (int i = 0; i < freq.size(); i++)
        D[i] = R * U[i];
    //for (int i = 0; i < 1; i++){
    //    std::cout.precision(3);
    //    std::cout << D[i] << std::endl;
    //}
    U.clear(); U.shrink_to_fit();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout.precision(5);
    std::cout << "The time for forward modeling is: " << ms_double.count() / 1000 << "s." << std::endl;

    return 1;
}

