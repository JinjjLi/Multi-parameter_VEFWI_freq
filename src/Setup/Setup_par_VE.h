#pragma once
#ifndef SETUP_PAR_VE_H_
#define SETUP_PAR_VE_H_
#endif 

#include "../General/includefile.h" 
#include "Make_General_source.h"
#include "Define_MC_point_receivers.h"
#include "Set_Acq.h"                                                                                                                                                
#include "Make_model.h"
#include "Make_P_sparse_alt.h"


class Setup_par_VE
{
    public:

        int nx; int nz; int PML_thick;
        int NN; int NPML; 
        int nxPML; int nzPML;
        int dx; int dz;
        int modeltype;
        float Amp_scale;
        int soffset, roffset;
        int sAcq, rAcq;
        int ns_sur, ns_SWD, ns;
        int nP;
        int P_grid, P_smooth;
        int numbands, step;
        float startband, endband1, endbandend;
        float omega0;
        float f0;
        
        Eigen::SparseMatrix<float> P;
        Eigen::SparseMatrix<float> P_big;

        Eigen::MatrixXf model0;
        Eigen::MatrixXf model_true;
        
        Eigen::RowVectorXf sx_sur, sx_SWD, sz_sur, sz_SWD;                                                                                                             
        Eigen::RowVectorXf rx, rz;

        std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf> sx;
        std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf> sz;
	    Eigen::RowVectorXf M11_sur;
	    Eigen::RowVectorXf M12_sur;
	    Eigen::RowVectorXf M22_sur;

        Eigen::SparseMatrix<float> S_sur;
        Eigen::SparseMatrix<float> dS_sur;
        Eigen::SparseMatrix<float> R;

        Eigen::RowVectorXf freq;
        Eigen::MatrixXcf fwave;

        std::tuple<Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf, Eigen::RowVectorXf> ind; 

        Setup_par_VE();
        virtual ~Setup_par_VE();

        void Get_model();
        void Read_model();
        void Read_par();
        void Get_Acq();

    private:

        void Get_select_P();
        void Get_freqs();
        void Get_inds();

};

