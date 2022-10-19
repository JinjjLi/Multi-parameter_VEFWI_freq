#pragma once
#ifndef FORWARD_VE_H_
#define FORWARD_VE_H_
#endif 

#include "../General/includefile.h" 
#include "../Setup/Setup_par_VE.h"
#include "Get_data_anelastic.h"


class Forward_VE
{
    public:
        Forward_VE(Setup_par_VE* this_par);
        virtual ~Forward_VE();

        void Get_D();

        std::vector<Eigen::SparseMatrix<std::complex<float>>> D;

    private:
        Setup_par_VE* this_par_alias;

};

