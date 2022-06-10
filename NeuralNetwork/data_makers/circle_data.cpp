#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

int main(){
    double x = 0;
    double y = 0;
    std::ofstream fout("circle_data.txt");
    for(int i = 0; i<20;i++){
        x = 0.8*cos(M_PI*i/10);
        y = 0.8*sin(M_PI*i/10);
        fout << x << '\t' << y << std::endl;
    }
    fout.close();
}