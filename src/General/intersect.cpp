#include "includefile.h"
#include "intersect.h"

std::vector<int> intersect(std::vector<int> &v1, std::vector<int> &v2){
    std::vector<int> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(),v1.end(),
                          v2.begin(),v2.end(),
                          back_inserter(v3));
    return v3;
}

