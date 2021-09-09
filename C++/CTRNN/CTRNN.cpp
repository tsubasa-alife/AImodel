#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include "matrix.h"

#define Input 2
#define Hidden 10
#define Output 2
#define tau 2.0

#define data_num 1
#define alpha 0.01
#define wb_width 1.0

#define data_size 20
#define n_time 19
#define epoch 10000

double get_rand(void);
Matrix activate(Matrix u, int column, int row);

class CTRNN{
    public:
        Matrix i2h;
        Matrix h2o;
        Matrix b_h;
        Matrix b_o;
        Matrix h2h;
        Matrix x;
        Matrix u_hid;
        Matrix c;
        Matrix c_prev;
        Matrix u_out;
        Matrix z;
        Matrix delta_i2h;
        Matrix grad_h2o;
        Matrix grad_b_o;
        Matrix grad_c;
        Matrix grad_i2h;
        Matrix grad_b_h;
        Matrix grad_h2h;

        CTRNN(){
            i2h = Matrix(Input,Hidden);
            h2o = Matrix(Hidden,Output);
            b_h = Matrix(data_num,Hidden);
            b_o = Matrix(data_num,Output);
            h2h = Matrix(Hidden,Hidden);
            x = Matrix(data_num,Input);
            u_hid = Matrix(data_num,Hidden);
            c = Matrix(data_num,Hidden);
            c_prev = Matrix(data_num,Hidden);
            u_out = Matrix(data_num,Output);
            z = Matrix(data_num,Output);
            delta_i2h = Matrix(data_num,Hidden);
            grad_h2o = Matrix(Hidden,Output);
            grad_b_o = Matrix(data_num,Output);
            grad_c = Matrix(data_num,Hidden);
            grad_i2h = Matrix(Input,Hidden);
            grad_b_h = Matrix(data_num,Hidden);
            grad_h2h = Matrix(Hidden,Hidden);
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

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    h2h[i][j]=wb_width * get_rand();
                    std::cout << "h2h[" << i << "]["<< j <<"] = "  << h2h[i][j] << std::endl;
                }
        }

        void forward(Matrix input){
            x = input;
            c_prev = c;
            u_hid = (1.0-(1.0/tau)) * u_hid + (1.0/tau) * (x*i2h + c_prev*h2h + b_h);
            c = activate(u_hid,data_num,Hidden);
            u_out = c*h2o + b_o;
            z = activate(u_out,data_num,Output);

        }

        void backward(Matrix input, Matrix output, Matrix context, Matrix context_prev, Matrix target){
            Matrix delta_h2o(data_num,Output);
            Matrix temp_z(data_num,Output);
            Matrix temp_c(data_num,Hidden);
            x = input;
            z = output;
            c = context;
            c_prev = context_prev;

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    temp_z[i][j]=1.0;
                }

             for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    temp_c[i][j]=1.0;
                }

            delta_h2o = ( z - target ) & (temp_z - (z & z));
            
            grad_h2o = grad_h2o + c.transposed() * delta_h2o;
            grad_b_o = grad_b_o + delta_h2o;
            grad_c = delta_h2o * h2o.transposed() + (1.0/tau) * delta_i2h * h2h.transposed();

            delta_i2h = (grad_c & (temp_c - (c & c))) + (1.0-(1.0/tau)) * delta_i2h;

            grad_i2h = grad_i2h + (1.0/tau) * x.transposed() * delta_i2h;
            grad_b_h = grad_b_h + (1.0/tau) * delta_i2h;
            grad_h2h = grad_h2h + (1.0/tau) * c_prev.transposed() * delta_i2h;
        }

        void update(){
            h2o = h2o - (alpha * grad_h2o);
            b_o = b_o - (alpha * grad_b_o);
            i2h = i2h - (alpha * grad_i2h);
            b_h = b_h - (alpha * grad_b_h);
            h2h = h2h - (alpha * grad_h2h);
        }

        void reset_state(){
            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    u_hid[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c_prev[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    u_out[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    z[i][j]=0.0;
                }
        }

        void clear_grads(){
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

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_h2h[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    delta_i2h[i][j]=0.0;
                }

        }


};

int main(){
    CTRNN model;
    Matrix data(20,2);
    Matrix x(1,2);
    Matrix target(1,2);
    Matrix output(1,2);
    Matrix context(1,10);
    Matrix context_prev(1,10);
    Matrix c_list(data_size,Hidden);
    Matrix z_list(n_time,Output);
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
        for(int t=1;t<=n_time;t++){

            x[1][1] = data[t][1];
            x[1][2] = data[t][2];
            target[1][1] = data[t+1][1];
            target[1][2] = data[t+1][2];
            
            model.forward(x);

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    c_list[t+1][j] = model.c[i][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Output;j++){
                    z_list[t][j] = model.z[i][j];
                }
            

        }

        for(int t=n_time;t>=1;t--){

            x[1][1] = data[t][1];
            x[1][2] = data[t][2];
            target[1][1] = data[t+1][1];
            target[1][2] = data[t+1][2];
            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    context[i][j] = c_list[t+1][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    context_prev[i][j] = c_list[t][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Output;j++){
                    output[i][j] = z_list[t][j];
                }

            model.backward(x,output,context,context_prev,target);

            total_loss += ((model.z[1][1] - target[1][1])*(model.z[1][1] - target[1][1]) + (model.z[1][2] - target[1][2])*(model.z[1][2] - target[1][2]))/2;
        }


        model.update();
        model.reset_state();
        model.clear_grads();

        std::cout << " total_loss:" << total_loss << std::endl;
    }

    std::ofstream fout("result.txt");
    for(int i = 1; i<=n_time;i++){
        fout << z_list[i][1] << '\t' << z_list[i][2] << std::endl;
    }
    fout.close();

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