#pragma once
#ifndef INVERSION_VE_H_
#define INVERSION_VE_H_
#endif 

#include "../General/includefile.h" 
#include "../Setup/Setup_par_VE.h"
#include "../Forward/Forward_VE.h"
#include "FDFWI_VE.h"

class Inversion_VE
{
    public:

        Inversion_VE(Setup_par_VE* this_par, Forward_VE* this_Forward);
        virtual ~Inversion_VE();

        void Set_start_model();
        void Set_ssmodel();
        void Set_inv_pars();
        void Read_inv_pars();
        void Run_inversion();


    private:

        int optype;
        int numits;
        int maxits;
        float tol;
        float reg_fac;
        float stabregfac;

        float scalem[5] = {0, 0, 0, 0, 0};
        float scaleS;
        float scaleSM;

        Setup_par_VE* this_par_alias;
        Forward_VE* this_Forward_alias;

        std::tuple<float *, float, float, float> scale;

        Eigen::MatrixXf ss_model0_inv;
        Eigen::MatrixXf model_start;

};

