#include "../General/includefile.h"
#include "cubint.h"

float cubint(float a, float b, float phia, float phib, float ga, float gb){
    float alpha = 0.0;
    if (a == b)
        alpha = a;
    else{
        float d1 = ga + gb - 3 * (phia - phib) / (a - b);
        float d2 = std::pow((std::pow(d1, 2.0) - ga * gb), 0.5);
        alpha = b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2));
    }
    return alpha;
}

