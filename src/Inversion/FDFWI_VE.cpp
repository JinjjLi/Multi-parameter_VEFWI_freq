#include "../General/includefile.h"
#include "FDFWI_VE.h"
#include "VE_Gradient.h"
#include "VE_Hvprod.h"
#include "TGN_optimization.h"

void FDFWI_VE(std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::RowVectorXf& freq, int step, Eigen::MatrixXcf& fwave, int nz, int nx, \
                     int dz, Eigen::MatrixXf& model, Eigen::MatrixXf& ssmodel0, float omega0, Eigen::SparseMatrix<float>& R, \
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, int optype, int numits, \
                     int PML_thick, float tol, int maxits, std::tuple<float *, float, float, float>& scale, \
                     std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, \
                     float reg_fac, float stabregfac, Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big){

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    Eigen::RowVectorXf sx_sur, sx_SWD, sz_sur, sz_SWD;
    std::tie(sx_sur, sx_SWD) = sx;
    std::tie(sz_sur, sz_SWD) = sz;
    int floorN = std::floor(freq.size() / step);

    for (int n = 0; n < floorN; n++){
        auto t1 = high_resolution_clock::now();
        Eigen::RowVectorXf subfreq = freq.block(0, n * step, 1, step);
        Eigen::MatrixXcf subfwave = fwave.block(n * step, 0, step, fwave.cols());
        std::vector<Eigen::SparseMatrix<std::complex<float>>> subD = std::vector<Eigen::SparseMatrix<std::complex<float>>>(D.begin() + n * step, D.begin() + n * step + step);
        for (int m = 0; m < numits; m++){
            if (optype == 2){
                auto t1 = high_resolution_clock::now();
                std::cout << "Starting gradient." << std::endl;
                float f; Eigen::MatrixXf g;
                std::tie(std::ignore, g) = VE_Gradient(subfreq, subfwave, omega0, model, \
                                                        ssmodel0, subD, R, sz, sx, MT_sur, \
                                                        nz, nx, dz, PML_thick, ind, scale, P);
                std::cout << "Gradient done." << std::endl;
                std::cout << "Starting optimization." << std::endl;
                TGN_optimization(model, ssmodel0, g, subfreq, subD, omega0, nz, nx, dz, PML_thick, R, \
                                 sz, sx, MT_sur, subfwave, ind, scale, P, P_big, tol, maxits, \
                                 reg_fac, stabregfac);
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                std::cout.precision(5);
                std::cout << "=== Finished " << n + 1 << "th outer iteration." << std::endl;
                std::cout << "The time for " << n + 1 << "th outer iteration: " << ms_double.count() / 1000 << "s." << std::endl;
                g.resize(0, 0);
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        std::cout.precision(5);
        std::cout << "=== Finished " << n + 1 << "th freqband." << std::endl;
        std::cout << "The time for " << n + 1 << "th freqband's inversion is: " << ms_double.count() / 1000 << "s." << std::endl;

        subfreq.resize(0);
        subfwave.resize(0, 0);
        subD.clear(); subD.shrink_to_fit();
    }                      
    sx_sur.resize(0); sx_SWD.resize(0);
    sz_sur.resize(0); sz_SWD.resize(0);
}


