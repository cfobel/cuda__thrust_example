#include <iostream>
#include "CudaInfo.hpp"
using namespace std;
using namespace cuda_info;

int main(int argc, char** argv) {
    CudaInfo ci;

    cout << ci.get_info() << endl;
    for(int i = 0; i < ci.get_device_count(); i++) {
        CudaDevice dev(i);
        cout << dev.get_device_info() << endl;
    }

    return 0;
}
