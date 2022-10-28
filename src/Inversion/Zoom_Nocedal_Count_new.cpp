#include "../General/includefile.h"
#include "Zoom_Nocedal_Count_new.h"
#include "cubint.h"
#include "VE_Gradient.h"

float Zoom_Nocedal_Count_new(Eigen::MatrixXf &x0, Eigen::MatrixXf& descentd, \
                             float c1, float c2, float phi0, float g0, \
                             float alphalo, float alphahi, float philo, float phihi, \
                             float glo, float ghi, Eigen::RowVectorXf& freq, \
                             Eigen::MatrixXcf& fwave, float omega0, Eigen::MatrixXf& ssmodel0, \
                             std::vector<Eigen::SparseMatrix<std::complex<float>>>& D, Eigen::SparseMatrix<float>& R, \
                             Eigen::SparseMatrix<float>& S, \
                             int nz, int nx, int dz, int PML_thick, \
                             std::tuple<float *, float, float, float>& scale, Eigen::SparseMatrix<float>& P){
    int i = 0;
    bool found = 0;
    float alpha = 0.0;
    float phi = 0.0;
    float g = 0.0;
    float alphastar = 0.0;
    Eigen::MatrixXf grad;
    Eigen::MatrixXf x0temp(x0.rows(), x0.cols());
    while (i < 10){
        if (alphalo < alphahi)
            alpha = cubint(alphalo, alphahi, philo, phihi, glo, ghi);
        else if (alphalo == alphahi)
            alpha = alphalo;
        else
            alpha = cubint(alphahi, alphalo, phihi, philo, ghi, glo);
        x0temp = x0 + alpha * descentd;
        std::tie(phi, grad) = VE_Gradient(freq, fwave, omega0, x0temp, \
                                                 ssmodel0, D, R, S, \
                                                 nz, nx, dz, PML_thick, scale, P);
        g = (grad.transpose() * descentd).value();
        if (phi > phi0 + c1 * alpha * g0 || phi >= philo){
            alphahi = alpha;
            phihi = phi;
            ghi = g;
        }
        else{
            if (std::abs(g) <= -c2 * g0){
                alphastar = alpha;
                found = 1;
                break;
            }
            if (g * (alphahi - alphalo) >= 0){
                alphahi = alphalo;
                phihi = philo;
                ghi = glo;
            }
            alphalo = alpha;
            philo = phi;
            glo = g;
        }
        i++;
        if (alphalo == alphahi){
            alphastar = alphalo;       
            break;
        }
    }

    if (!found){
        if (alphalo < alphahi)
            alphastar = cubint(alphalo, alphahi, philo, phihi, glo, ghi);
        else
            alphastar = cubint(alphahi, alphalo, phihi, philo, ghi, glo);
    }

    grad.resize(0, 0); 
    x0temp.resize(0, 0);
    return alphastar;
}

