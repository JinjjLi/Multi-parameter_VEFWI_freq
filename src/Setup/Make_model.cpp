#include "../General/includefile.h"
#include "Make_model.h"

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> Make_model(int nz, int nx, int type)
{
    Eigen::MatrixXf model0 = Eigen::MatrixXf::Constant(nz * nx, 5, 0.0);
    Eigen::MatrixXf vp_0 = Eigen::MatrixXf::Constant(nz, nx, 2700.0);
    Eigen::MatrixXf vs_0 = Eigen::MatrixXf::Constant(nz, nx, 1600.0);
    Eigen::MatrixXf rho_0 = Eigen::MatrixXf::Constant(nz, nx, 1400.0);
    Eigen::MatrixXf Qp_inv_0 = Eigen::MatrixXf::Constant(nz, nx, 0.01);
    Eigen::MatrixXf Qs_inv_0 = Eigen::MatrixXf::Constant(nz, nx, 0.02);
    //Eigen::MatrixXf c11(nz, nx), c44(nz, nx);
    //c11 = vp_true.array() * rho_true.array().pow(2.0);
    //c44 = vs_true.array() * rho_true.array().pow(2.0);
    
    rho_0.resize(nz * nx, 1); vp_0.resize(nz * nx, 1); vs_0.resize(nz * nx, 1);
    Qp_inv_0.resize(nz * nx, 1); Qs_inv_0.resize(nz * nx, 1);
    
    model0.col(0) = rho_0.array();
    model0.col(1) = vp_0.array(); model0.col(2) = Qp_inv_0.array();
    model0.col(3) = vs_0.array(); model0.col(4) = Qs_inv_0.array();
   
    int center_x = 15; int center_z = 15; int R = 4;
    
    Eigen::MatrixXf model_true = Eigen::MatrixXf::Constant(nz * nx, 5, 0.0);
    
    switch(type){
    case 1:{
               Eigen::MatrixXf vp_true = vp_0;
               vp_true.resize(nz, nx);
               for (int i = 0; i < nz; i++)
                   for (int j = 0; j < nx; j++){
                       if (sqrt(pow(i - center_z, 2) + pow(j - center_x, 2)) < R)
                           vp_true(i, j) = 3300.0;
                   }
               vp_true.resize(nz * nx, 1); 
               model_true.col(0) = rho_0.array(); model_true.col(1) = vp_true.array(); model_true.col(2) = Qp_inv_0.array();
               model_true.col(3) = vs_0.array(); model_true.col(4) = Qs_inv_0.array();
               vp_true.resize(0, 0);
           }
           break;
    case 2:{
               Eigen::MatrixXf vp_true = vp_0; Eigen::MatrixXf rho_true = rho_0;
               vp_true.resize(nz, nx); rho_true.resize(nz, nx);
               for (int i = 0; i < nz; i++)
                   for (int j = 0; j < nx; j++){
                       if (sqrt(pow(i - center_z, 2) + pow(j - center_x, 2)) < R){
                           vp_true(i, j) = 3300.0; rho_true(i, j) = 1900.0;
                       }
                   }
               vp_true.resize(nz * nx, 1); rho_true.resize(nz * nx, 1);
               model_true.col(0) = rho_true.array(); model_true.col(1) = vp_true.array(); model_true.col(2) = Qp_inv_0.array();
               model_true.col(3) = vs_0.array(); model_true.col(4) = Qs_inv_0.array();
               vp_true.resize(0, 0); rho_true.resize(0, 0);
           }      
           break;
   }
    
    vp_0.resize(0, 0); rho_0.resize(0, 0); vs_0.resize(0, 0);
    Qp_inv_0.resize(0, 0); Qs_inv_0.resize(0, 0);

    model0.resize(nz * nx * 5, 1);
    model_true.resize(nz * nx * 5, 1);
    std::cout << "Model done." << std::endl;
    return std::make_tuple(model0, model_true);
}

