#include "../Setup/Setup_par_VE.h"
#include "../Forward/Forward_VE.h"
#include "../Inversion/Inversion_VE.h"

int main(){                                                                                                 

    Setup_par_VE* this_par = new Setup_par_VE();
    Forward_VE* this_forward = new Forward_VE(this_par);
    this_forward->Get_D();
    Inversion_VE* this_inv = new Inversion_VE(this_par, this_forward);
    this_inv->Set_start_model();
    this_inv->Set_ssmodel();
    this_inv->Read_inv_pars();
    //this_inv->Set_inv_pars();
    this_inv->Run_inversion();

    delete this_inv;
    delete this_forward;
    delete this_par;

    return 0;

}
