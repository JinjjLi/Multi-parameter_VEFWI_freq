#include "Inversion_VE.h"

Inversion_VE::Inversion_VE(Setup_par_VE* this_par, Forward_VE* this_Forward){

    this_par_alias = this_par;
    this_Forward_alias = this_Forward;

    scalem[0] = this_par_alias->model_true.block(0, 0, this_par_alias->NN, 1).maxCoeff();
    scalem[1] = this_par_alias->model_true.block(this_par_alias->NN, 0, this_par_alias->NN, 1).maxCoeff();
    scalem[2] = this_par_alias->model_true.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1).maxCoeff();
    scalem[3] = this_par_alias->model_true.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1).maxCoeff();
    scalem[4] = this_par_alias->model_true.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1).maxCoeff();
    scaleS = 0; 
    scaleSM = 1;

    scale = std::make_tuple(scalem, scaleS, 1.0, scaleSM);

    Set_start_model();
    Set_ssmodel();
    Read_inv_pars();
}

Inversion_VE::~Inversion_VE(){

    ss_model0_inv.resize(0, 0);
    model_start.resize(0, 0);

    std::cout << "Inv object destroyed." << std::endl;

}


void Inversion_VE::Set_start_model(){

    model_start = Eigen::MatrixXf(this_par_alias->P.cols(), 1);
    model_start.setZero();

}

void Inversion_VE::Set_ssmodel(){

    ss_model0_inv = Eigen::MatrixXf(this_par_alias->model0.rows(), this_par_alias->model0.cols());                                                             
    ss_model0_inv.block(0, 0, this_par_alias->NN, 1) = this_par_alias->model0.block(0, 0, this_par_alias->NN, 1).array() / scalem[0];
    ss_model0_inv.block(this_par_alias->NN, 0, this_par_alias->NN, 1) = this_par_alias->model0.block(this_par_alias->NN, 0, this_par_alias->NN, 1).array() / scalem[1];
    ss_model0_inv.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1) = this_par_alias->model0.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() / scalem[2];
    ss_model0_inv.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1) = this_par_alias->model0.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() / scalem[3];
    ss_model0_inv.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1) = this_par_alias->model0.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() / scalem[4];

}

void Inversion_VE::Read_inv_pars(){

    std::ifstream f1;
    f1.open("./data/inv_par.txt");
    if(!f1){
        std::cout << "Error reading inversion parameters!!!" << std::endl;
        exit(0);
    }
    while(!f1.eof()){
        f1 >> optype;
        f1 >> numits;
        f1 >> maxits;
        f1 >> tol;
        f1 >> reg_fac;
    }
    std::cout << "Reading inv par over." << std::endl;
    f1.close();
}

void Inversion_VE::Set_inv_pars(){

    optype = 2;
    numits = 2;
    maxits = 20;
    tol = 1e-5;;
    reg_fac = 1e-2;
    stabregfac = reg_fac;
}

void Inversion_VE::Run_inversion(){

        std::cout << "Inversion starts." << std::endl;
        FDFWI_VE(this_Forward_alias->D, this_par_alias->freq, this_par_alias->step, this_par_alias->fwave, this_par_alias->nz, this_par_alias->nx, \
                 this_par_alias->dz, model_start, ss_model0_inv, \
                 this_par_alias->omega0, this_par_alias->R, this_par_alias->S, \
                 optype, numits, this_par_alias->PML_thick, tol, \
                 maxits, scale, reg_fac, stabregfac, this_par_alias->P, this_par_alias->P_big);

        Eigen::MatrixXf model_out = this_par_alias->P * model_start + ss_model0_inv;
        model_out.block(0, 0, this_par_alias->NN, 1) = model_out.block(0, 0, this_par_alias->NN, 1).array() * scalem[0];
        model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1).array() * scalem[1];
        model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1).array().pow(-0.5);
        model_out.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() * scalem[2];
        model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() * scalem[3];
        model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1).array().pow(-0.5);
        model_out.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1) = model_out.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1).array() * scalem[4];


        std::ofstream invmodel("./data/invmodel.dat", std::ios::out | std::ios::trunc);
        if(invmodel){
            switch(this_par_alias -> nP){
            case 1:{ 
                       Eigen::MatrixXf rho_out = model_out.block(0, 0, this_par_alias->NN, 1);
                       invmodel << rho_out << "\n";
                       rho_out.resize(0, 0);
                   }
                   break;
            case 2:{ 
                       Eigen::MatrixXf rho_out = model_out.block(0, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vp_out = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1);
                       invmodel << rho_out << "\n" << vp_out << "\n";
                       rho_out.resize(0, 0); vp_out.resize(0, 0);
                   }
                   break;
            case 3:{ 
                       Eigen::MatrixXf rho_out = model_out.block(0, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vp_out = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf Qp_out = model_out.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       invmodel << rho_out << "\n" << vp_out << "\n" << Qp_out << "\n";
                       rho_out.resize(0, 0); vp_out.resize(0, 0); Qp_out.resize(0, 0);
                   }
                   break;
            case 4:{ 
                       Eigen::MatrixXf rho_out = model_out.block(0, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vp_out = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf Qp_out = model_out.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vs_out = model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       invmodel << rho_out << "\n" << vp_out << "\n" << Qp_out << "\n" << vs_out <<"\n";
                       rho_out.resize(0, 0); vp_out.resize(0, 0); Qp_out.resize(0, 0); vs_out.resize(0, 0);
                   }
                   break;
            case 5:{ 
                       Eigen::MatrixXf rho_out = model_out.block(0, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vp_out = model_out.block(this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf Qp_out = model_out.block(2 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf vs_out = model_out.block(3 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       Eigen::MatrixXf Qs_out = model_out.block(4 * this_par_alias->NN, 0, this_par_alias->NN, 1);
                       invmodel << rho_out << "\n" << vp_out << "\n" << Qp_out << "\n" << vs_out <<"\n" << Qs_out <<"\n";
                       rho_out.resize(0, 0); vp_out.resize(0, 0); Qp_out.resize(0, 0); vs_out.resize(0, 0); Qs_out.resize(0, 0);
                   }
                   break;
            }
        invmodel.close();
        }
        //std::ofstream truemodel("./data/truemodel.dat", std::ios::out | std::ios::trunc);
        //if(truemodel){
        //    truemodel << vp_true << "\n" << rho_true << "\n";
        //    truemodel.close();
        //}
        //std::ofstream invmodel("./data/invmodel.dat", std::ios::out | std::ios::trunc);
        //if(invmodel){
        //    invmodel << model_out << "\n";
        //    invmodel.close();
        //}

        model_out.resize(0, 0); 

        std::cout << "Inversion ended." << std::endl;

}
