#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include "matrix.h"

#define Input 2
#define Hidden 10
#define Output 2

#define data_num 1
#define alpha 0.1
#define wb_width 1.0

#define data_size 20
#define epoch 10

double get_rand(void);
Matrix activate(Matrix u, int column, int row);

class FNN{
    public:
        Matrix i2h;
        Matrix h2o;
        Matrix b_h;
        Matrix b_o;
        Matrix x;
        Matrix u_hid;
        Matrix c;
        Matrix u_out;
        Matrix z;
        Matrix grad_h2o;
        Matrix grad_b_o;
        Matrix grad_c;
        Matrix grad_i2h;
        Matrix grad_b_h;

        FNN(){
            i2h = Matrix(Input,Hidden);
            h2o = Matrix(Hidden,Output);
            b_h = Matrix(data_num,Hidden);
            b_o = Matrix(data_num,Output);
            x = Matrix(data_num,Input);
            u_hid = Matrix(data_num,Hidden);
            c = Matrix(data_num,Hidden);
            u_out = Matrix(data_num,Output);
            z = Matrix(data_num,Output);
            grad_h2o = Matrix(Hidden,Output);
            grad_b_o = Matrix(data_num,Output);
            grad_c = Matrix(data_num,Hidden);
            grad_i2h = Matrix(Input,Hidden);
            grad_b_h = Matrix(data_num,Hidden);
        }

        void init(){
            for(int i=1;i<=Input;i++)
                for(int j=1;j<=Hidden;j++){
                    i2h[i][j]=wb_width * get_rand();
                    std::cout << "i2h[" << i << "]["<< j <<"] = "  << i2h[i][j] << std::endl;
                }
            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Output;j++){
                    h2o[i][j]=wb_width * get_rand();
                    std::cout << "h2o[" << i << "]["<< j <<"] = "  << h2o[i][j] << std::endl;
                }
            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    b_h[i][j]=wb_width * get_rand();
                    std::cout << "b_h[" << i << "]["<< j <<"] = "  << b_h[i][j] << std::endl;
                }
            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    b_o[i][j]=wb_width * get_rand();
                    std::cout << "b_o[" << i << "]["<< j <<"] = "  << b_o[i][j] << std::endl;
                }
        }

        void forward(Matrix input){
            x = input;
            u_hid = x*i2h + b_h;
            c = activate(u_hid,data_num,Hidden);
            u_out = c*h2o + b_o;
            z = activate(u_out,data_num,Output);

        }

        void backward(Matrix target){
            Matrix delta_h2o(data_num,Output);
            Matrix delta_i2h(data_num,Hidden);
            Matrix temp_z(data_num,Output);
            Matrix temp_c(data_num,Hidden);
            Matrix unchi(data_num,Output);

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    temp_z[i][j]=1.0;
                }

             for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    temp_c[i][j]=1.0;
                }

            delta_h2o = ( z - target ) & (temp_z - (z & z));
            
            grad_h2o = c.transposed() * delta_h2o;
            grad_b_o = delta_h2o;
            grad_c = delta_h2o * h2o.transposed();

            delta_i2h = grad_c & (temp_c - (c & c));

            grad_i2h = x.transposed() * delta_i2h;
            grad_b_h = delta_i2h;
        }

        void update(){
            h2o = h2o - (alpha * grad_h2o);
            b_o = b_o - (alpha * grad_b_o);
            i2h = i2h - (alpha * grad_i2h);
            b_h = b_h - (alpha * grad_b_h);
        }

        void cleargrads(){
            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Output;j++){
                    grad_h2o[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    grad_b_o[i][j]=0.0;
                }

            for(int i=1;i<=Input;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_i2h[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_b_h[i][j]=0.0;
                }

        }


};

int main(){
    FNN model;
    Matrix data(20,2);
    Matrix x(1,2);
    Matrix t(1,2);
    Matrix test(1,2);
    double total_loss = 0;

    std::ifstream fin("data.txt");
    for(int i=1;i<=data_size;i++)
        for(int j=1;j<=Input;j++){
            fin >> data[i][j];
            std::cout << "data[" << i << "]["<< j <<"] = "  << data[i][j] << std::endl;
        }

    fin.close();

    model.init();

    for(int i=1;i<=epoch;i++){
        total_loss = 0;
        std::cout << "epoch:" << i << std::endl;
        for(int s=1;s<=data_size-1;s++){

            x[1][1] = data[s][1];
            x[1][2] = data[s][2];
            t[1][1] = data[s+1][1];
            t[1][2] = data[s+1][2];
            
            model.forward(x);
            model.backward(t);
            model.update();
            
            total_loss += ((model.z[1][1] - t[1][1])*(model.z[1][1] - t[1][1]) + (model.z[1][2] - t[1][2])*(model.z[1][2] - t[1][2]))/2;

        }

        std::cout << " total_loss:" << total_loss << std::endl;
    }

}

Matrix activate(Matrix u, int column, int row){
    Matrix temp(column,row);
    for(int i=1;i<=column;i++)
        for(int j=1;j<=row;j++){
            temp[i][j]=std::tanh(u[i][j]);
        }
    return temp;
}

double get_rand(void){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(0.0,0.01);

    return dist(mt);
}