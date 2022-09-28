#include "../General/includefile.h"
#include "Linesearch_Nocedal_Full_new.h"
#include "VE_Gradient.h"
#include "Zoom_Nocedal_Count_new.h"

float Linesearch_Nocedal_Full_new(Eigen::MatrixXf &x0, Eigen::MatrixXf& descentd, \
                                  float c1, float c2, float alpha0, Eigen::RowVectorXf& freq, \
                                  Eigen::MatrixXcf& fwave, float omega0, \
                                  Eigen::MatrixXf& ssmodel0, std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::SparseMatrix<float>& R, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sz, std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf>& sx, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& MT_sur, \
                                  int nz, int nx, int dz, int PML_thick, \
                                  std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf>& ind, \
                                  std::tuple<float *, float, float, float>& scale, Eigen::SparseMatrix<float>& P){
    float alpha = alpha0;
    float alphaprev = 0.0;
    float phi0; 
    Eigen::MatrixXf grad0;
    std::tie(phi0, grad0) = VE_Gradient(freq, fwave, omega0, x0, \
                                              ssmodel0, D, R, sz, sx, MT_sur, \
                                              nz, nx, dz, PML_thick, ind, scale, P);
    std::cout << "Initial objective function is: " << phi0 << std::endl;
    float phiprev = phi0;
    float g0 = (grad0.transpose() * descentd).value();
    float gprev = g0;
    float alphastar;
    Eigen::MatrixXf grad;
    Eigen::MatrixXf x0temp;
    float phi;
    if(g0 < 0){
        int i = 0; 
        bool found = 0;
        alphastar = 0.0;
        while (i < 10){
            x0temp = x0 + alpha * descentd;
            std::tie(phi, grad) = VE_Gradient(freq, fwave, omega0, x0temp, \
                                                     ssmodel0, D, R, sz, sx, MT_sur, \
                                                     nz, nx, dz, PML_thick, ind, scale, P);
            int whileit = 0;
            while (isinf(phi)){
                whileit++;
                alpha /= 2.0;
                x0temp = x0 + alpha * descentd;
                std::tie(phi, grad) = VE_Gradient(freq, fwave, omega0, x0temp, \
                                                         ssmodel0, D, R, sz, sx, MT_sur, \
                                                         nz, nx, dz, PML_thick, ind, scale, P);
                if (whileit >= 10)
                    phi = 0.0;
            }
            if (whileit >= 10)
                break;
            float g = (grad.transpose() * descentd).value();
            std::cout << "g: " << g << std::endl;
            if (phi > phi0 + c1 * alpha * g0 || (phi >= phiprev && i > 0)){
                std::cout << "phi > phi0 + c1 * alpha * g0 || (phi >= phiprev && i > 0)" << std::endl;
                alphastar = Zoom_Nocedal_Count_new(x0, descentd, c1, c2, phi0, g0, alphaprev, \
                                                   alpha, phiprev, phi, gprev, g, \
                                                   freq, fwave, omega0, ssmodel0, D, R, \
                                                   sz, sx, MT_sur, nz, nx, dz, PML_thick, \
                                                   ind, scale, P);
                found = 1;
                break;
            }
            if (std::abs(g) <= -c2 * g0){
                std::cout << "g <= -c2 * g0." << std::endl;
                alphastar = alpha;
                found = 1;
                break;
            }
            if (g >= 0){
                std::cout << "g >= 0." << std::endl;
                alphastar = Zoom_Nocedal_Count_new(x0, descentd, c1, c2, phi0, g0, alphaprev, \
                                                   alpha, phiprev, phi, gprev, g, \
                                                   freq, fwave, omega0, ssmodel0, D, R, \
                                                   sz, sx, MT_sur, nz, nx, dz, PML_thick, \
                                                   ind, scale, P);
                found = 1;
                break;
            }
            alphaprev = alpha;
            gprev = g;
            alpha *= 2.0;
            phiprev = phi;
            i++;
        }
        if (!found)
            alphastar = 0.0;
    }
    else{
        alphastar = 0.0;
        phi = phi0;
    }
    x0temp = x0 + alphastar * descentd;
    std::tie(phi, std::ignore) = VE_Gradient(freq, fwave, omega0, x0temp, \
                                             ssmodel0, D, R, sz, sx, MT_sur, \
                                             nz, nx, dz, PML_thick, ind, scale, P);
    std::cout << "Objective function after line search is: " << phi << std::endl;
    grad0.resize(0, 0); grad.resize(0, 0);
    x0temp.resize(0, 0);
    return alphastar;
}

