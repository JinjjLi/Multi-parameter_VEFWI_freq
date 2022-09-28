#include "../General/includefile.h"
#include "LBFGS_solve_linear.h"
#include "General_VE_Hvprod.h"
#include "BFGSprod.h"

Eigen::MatrixXf LBFGS_solve_linear(Eigen::MatrixXf &model, Eigen::MatrixXf& ssmodel0, \
                                   Eigen::MatrixXf& b, Eigen::RowVectorXf& frequency, \
            float omega0, int nz, int nx, int dz, int PML_thick, Eigen::SparseMatrix<float>& R, \
            std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, \
            std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
            std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, \
            Eigen::MatrixXcf& fwave, float reg_fac, float stabregfac, \
            std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, \
            std::tuple<float *, float, float, float>& scale, \
            Eigen::SparseMatrix<float>& P, Eigen::SparseMatrix<float>& P_big, float tol, int maxits){

    Eigen::MatrixXf u = Eigen::MatrixXf(b.rows(), b.cols());
    u.setZero();
    Eigen::MatrixXf r = -b;
    Eigen::MatrixXf u_best = Eigen::MatrixXf(r.rows(), u.cols());
    u_best.setZero();
    float initnorm = b.norm();
    Eigen::MatrixXf ustore = Eigen::MatrixXf(u.rows(), maxits);
    Eigen::MatrixXf rstore = Eigen::MatrixXf(u.rows(), maxits);
    ustore.setZero();
    rstore.setZero();
    
    int n = 0; bool converged = 0;

    Eigen::MatrixXf p = Eigen::MatrixXf(b.rows(), b.cols());
    p.setZero();
    std::vector<float> resvec;
    float res_past = 0.0;
    Eigen::SparseMatrix<float> H0(u.rows(), u.rows());
    while (!converged){
        if (n < maxits){
            ustore.col(n) = u; rstore.col(n) = r;
        }
        else{
            ustore.leftCols(ustore.cols() - 1) = ustore.rightCols(ustore.cols() - 1);
            ustore.col(ustore.cols() - 1) = u;
            rstore.leftCols(rstore.cols() - 1) = rstore.rightCols(rstore.cols() - 1);
            rstore.col(rstore.cols() - 1) = r;
        }
        if (n >= 1 && n < maxits){
            Eigen::MatrixXf send = ustore.col(n) - ustore.col(n - 1);
            Eigen::MatrixXf yend = rstore.col(n) - rstore.col(n - 1);
            float ind = (send.transpose() * yend).value() / (yend.transpose() * yend).value();
            if ((yend.transpose() * yend).value() == 0.0)
                ind = 0.0;

            Eigen::MatrixXf partial_rstore = rstore.leftCols(n + 1);
            Eigen::MatrixXf partial_ustore = ustore.leftCols(n + 1);
            spdiags_noncomplex<float>(H0, Eigen::MatrixXf::Constant(u.rows(), u.cols(), ind), \
                                                               Eigen::MatrixXi::Zero(1, 1), \
                                                               u.rows(), u.rows());
            p = BFGSprod(r, partial_rstore, partial_ustore, H0);
            send.resize(0, 0); yend.resize(0, 0);
            //H0.resize(0, 0); H0.data().squeeze();  
            H0.setZero();
            partial_rstore.resize(0, 0);
            partial_ustore.resize(0, 0);
        }
        else if (n >= maxits){
            Eigen::MatrixXf send = ustore.col(maxits - 1) - ustore.col(maxits - 2);
            Eigen::MatrixXf yend = rstore.col(maxits - 1) - rstore.col(maxits - 2);
            float ind = (send.transpose() * yend).value() / (yend.transpose() * yend).value();
            if ((yend.transpose() * yend).value() == 0.0)
                ind = 0.0;
            spdiags_noncomplex<float>(H0, Eigen::MatrixXf::Constant(u.rows(), u.cols(), ind), \
                                                               Eigen::MatrixXi::Zero(1, 1), \
                                                               u.rows(), u.rows());
            p = BFGSprod(r, rstore, ustore, H0);
            send.resize(0, 0); yend.resize(0, 0);
            H0.setZero();
            //H0.resize(0, 0); H0.data().squeeze();
        }
        else 
            p = r;

        //Eigen::MatrixXf Hp = VE_Hvprod(model, ssmodel0, p, frequency, omega0, \
        //                                                  nz, nx, dz, PML_thick, R, sz, sx, \
        //                                                  MT_sur, fwave, reg_fac, stabregfac, \
        //                                                  ind, scale, P, P_big);
        Eigen::MatrixXf Hp = General_VE_Hvprod(model, ssmodel0, p, frequency, omega0, \
                                                          nz, nx, dz, PML_thick, R, sz, sx, \
                                                          MT_sur, fwave, reg_fac, stabregfac, \
                                                          ind, scale, P, P_big);
        float alpha = -(r.transpose().cast<double>() * p.cast<double>()).value() / \
                      (p.transpose().cast<double>() * Hp.cast<double>()).value();

        int red_it = 0;
        Eigen::MatrixXf mat = r + alpha * Hp;
        while (mat.norm() > r.norm()){
            alpha *= 0.9;
            red_it++;
            if (red_it >= 10)
                break;
        }
        u += alpha * p;
        r += alpha * Hp;
        float relres = r.norm() / initnorm;
        resvec.push_back(relres);
        if (relres == *std::min_element(resvec.begin(), resvec.end()))
            u_best = u;
        std::cout << "Relative residual: " << relres << std::endl;
        n++;
        if(relres < tol || n >= maxits - 1)
            converged = 1;   
        mat.resize(0, 0);
    }
        u.resize(0, 0); r.resize(0, 0); ustore.resize(0, 0); rstore.resize(0, 0);
        p.resize(0, 0); 
        H0.resize(0, 0); H0.data().squeeze();
        
        return u_best;
}

