#include "../General/includefile.h"
#include "TGN_optimization.h"
#include "LBFGS_solve_linear.h"
#include "Linesearch_Nocedal_Full_new.h"

void TGN_optimization(Eigen::MatrixXf& model, Eigen::MatrixXf& ssmodel0, Eigen::MatrixXf& g, Eigen::RowVectorXf& frequency, \
            std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, \
            float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
            Eigen::SparseMatrix<float>& S, \
            Eigen::MatrixXcf& fwave, \
            std::tuple<float *, float, float, float>& scale, \
            Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big, float tol, int maxits, \
            float reg_fac, float stabregfac){

    float Wolfe1 = 1e-4; float Wolfe2 = 0.9;
    Eigen::MatrixXf minus_g = -g;
    g.resize(0, 0);
    Eigen::MatrixXf descent_d = LBFGS_solve_linear(model, ssmodel0, minus_g, frequency, omega0, nz, nx, dz, PML_thick, R, \
                                                   S, fwave, reg_fac, stabregfac, \
                                                   scale, P, P_big, \
                                                   tol, maxits);
    float alpha0 = 1.0;
    float alpha = Linesearch_Nocedal_Full_new(model, descent_d, Wolfe1, Wolfe2, alpha0, frequency, fwave, \
                                              omega0, ssmodel0, D, R, S, nz, nx, dz, PML_thick, \
                                              scale, P);
    std::cout << "Alpha: " << alpha << std::endl;
    model += alpha * descent_d;

    descent_d.resize(0, 0);
    minus_g.resize(0, 0);
}

