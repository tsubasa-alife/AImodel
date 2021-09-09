#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include "matrix.h"

#define Input 2
#define Hidden 10
#define Output 2
#define tau_f 2.0
#define tau_s 5.0

#define data_num 1
#define alpha 0.01
#define wb_width 1.0

#define data_size 20
#define n_time 19
#define epoch 10000

double get_rand(void);
Matrix activate(Matrix u, int column, int row);

class MTRNN{
    public:
        Matrix i2h;
        Matrix h2o;
        Matrix b_h_f;
        Matrix b_o;
        Matrix h2h_f;
        Matrix b_h_s;
        Matrix h2h_s;
        Matrix s2f;
        Matrix f2s;
        Matrix x;
        Matrix u_hid_f;
        Matrix u_hid_s;
        Matrix c_f;
        Matrix c_f_prev;
        Matrix c_s;
        Matrix c_s_prev;
        Matrix u_out;
        Matrix z;
        Matrix delta_h_f;
        Matrix delta_h_s;
        Matrix grad_h2o;
        Matrix grad_b_o;
        Matrix grad_c_f;
        Matrix grad_c_s;
        Matrix grad_i2h;
        Matrix grad_b_h_f;
        Matrix grad_b_h_s;
        Matrix grad_h2h_f;
        Matrix grad_h2h_s;
        Matrix grad_s2f;
        Matrix grad_f2s;

        MTRNN(){
            i2h = Matrix(Input,Hidden);
            h2o = Matrix(Hidden,Output);
            b_h_f = Matrix(data_num,Hidden);
            b_o = Matrix(data_num,Output);
            h2h_f = Matrix(Hidden,Hidden);
            b_h_s = Matrix(data_num,Hidden);
            h2h_s = Matrix(Hidden,Hidden);
            s2f = Matrix(Hidden,Hidden);
            f2s = Matrix(Hidden,Hidden);
            x = Matrix(data_num,Input);
            u_hid_f = Matrix(data_num,Hidden);
            u_hid_s = Matrix(data_num,Hidden);
            c_f = Matrix(data_num,Hidden);
            c_f_prev = Matrix(data_num,Hidden);
            c_s = Matrix(data_num,Hidden);
            c_s_prev = Matrix(data_num,Hidden);
            u_out = Matrix(data_num,Output);
            z = Matrix(data_num,Output);
            delta_h_f = Matrix(data_num,Hidden);
            delta_h_s = Matrix(data_num,Hidden);
            grad_h2o = Matrix(Hidden,Output);
            grad_b_o = Matrix(data_num,Output);
            grad_c_f = Matrix(data_num,Hidden);
            grad_c_s = Matrix(data_num,Hidden);
            grad_i2h = Matrix(Input,Hidden);
            grad_b_h_f = Matrix(data_num,Hidden);
            grad_b_h_s = Matrix(data_num,Hidden);
            grad_h2h_f = Matrix(Hidden,Hidden);
            grad_h2h_s = Matrix(Hidden,Hidden);
            grad_s2f = Matrix(Hidden,Hidden);
            grad_f2s = Matrix(Hidden,Hidden);

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
                    b_h_f[i][j]=wb_width * get_rand();
                    std::cout << "b_h_f[" << i << "]["<< j <<"] = "  << b_h_f[i][j] << std::endl;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    b_h_s[i][j]=wb_width * get_rand();
                    std::cout << "b_h_s[" << i << "]["<< j <<"] = "  << b_h_s[i][j] << std::endl;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    b_o[i][j]=wb_width * get_rand();
                    std::cout << "b_o[" << i << "]["<< j <<"] = "  << b_o[i][j] << std::endl;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    h2h_f[i][j]=wb_width * get_rand();
                    std::cout << "h2h_f[" << i << "]["<< j <<"] = "  << h2h_f[i][j] << std::endl;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    h2h_s[i][j]=wb_width * get_rand();
                    std::cout << "h2h_s[" << i << "]["<< j <<"] = "  << h2h_s[i][j] << std::endl;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    s2f[i][j]=wb_width * get_rand();
                    std::cout << "s2f[" << i << "]["<< j <<"] = "  << s2f[i][j] << std::endl;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    f2s[i][j]=wb_width * get_rand();
                    std::cout << "f2s[" << i << "]["<< j <<"] = "  << f2s[i][j] << std::endl;
                }
        }

        void forward(Matrix input){
            x = input;
            c_f_prev = c_f;
            c_s_prev = c_s;
            u_hid_f = (1.0-(1.0/tau_f)) * u_hid_f + (1.0/tau_f) * (x*i2h + c_f_prev*h2h_f + c_s_prev*s2f + b_h_f);
            u_hid_s = (1.0-(1.0/tau_s)) * u_hid_s + (1.0/tau_s) * (c_s_prev*h2h_s + c_f_prev*f2s + b_h_s);
            c_f = activate(u_hid_f,data_num,Hidden);
            c_s = activate(u_hid_s,data_num,Hidden);
            u_out = c_f*h2o + b_o;
            z = activate(u_out,data_num,Output);

        }

        void backward(Matrix input, Matrix output, Matrix context_f, Matrix context_f_prev, Matrix context_s, Matrix context_s_prev, Matrix target){
            Matrix delta_h2o(data_num,Output);
            Matrix temp_z(data_num,Output);
            Matrix temp_c(data_num,Hidden);
            x = input;
            z = output;
            c_f = context_f;
            c_f_prev = context_f_prev;
            c_s = context_s;
            c_s_prev = context_s_prev;

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Output;j++){
                    temp_z[i][j]=1.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    temp_c[i][j]=1.0;
                }

            delta_h2o = ( z - target ) & (temp_z - (z & z));
            
            grad_h2o = grad_h2o + c_f.transposed() * delta_h2o;
            grad_b_o = grad_b_o + delta_h2o;
            grad_c_f = delta_h2o * h2o.transposed() + (1.0/tau_f) * delta_h_f * h2h_f.transposed();
            grad_c_s = (1.0/tau_f)*delta_h_f*s2f.transposed() + delta_h_s*h2h_s.transposed();

            delta_h_s = (grad_c_s & (temp_c - (c_s & c_s))) + (1.0-(1.0/tau_s)) * delta_h_s;
            /*for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    std::cout<< "delta:" << delta_h_s[i][j] <<std::endl;
                }*/
            delta_h_f = (grad_c_f & (temp_c - (c_f & c_f))) + (1.0-(1.0/tau_f)) * delta_h_f +(1.0/tau_s)*delta_h_s*f2s.transposed();

            grad_i2h = grad_i2h + (1.0/tau_f) * x.transposed() * delta_h_f;
            grad_b_h_f = grad_b_h_f + (1.0/tau_f) * delta_h_f;
            grad_h2h_f = grad_h2h_f + (1.0/tau_f) * c_f_prev.transposed() * delta_h_f;
            grad_b_h_s = grad_b_h_s + (1.0/tau_s) * delta_h_s;
            grad_h2h_s = grad_h2h_s + (1.0/tau_s) * c_s_prev.transposed() * delta_h_s;
            grad_s2f = grad_s2f + (1.0/tau_f) * c_s_prev.transposed() * delta_h_f;
            grad_f2s = grad_f2s + (1.0/tau_s) * c_f_prev.transposed() * delta_h_s;
        }

        void update(){
            h2o = h2o - (alpha * grad_h2o);
            b_o = b_o - (alpha * grad_b_o);
            i2h = i2h - (alpha * grad_i2h);
            b_h_f = b_h_f - (alpha * grad_b_h_f);
            h2h_f = h2h_f - (alpha * grad_h2h_f);
            b_h_s = b_h_s - (alpha * grad_b_h_s);
            h2h_s = h2h_s - (alpha * grad_h2h_s);
            s2f = s2f - (alpha * grad_s2f);
            f2s = f2s - (alpha * grad_f2s);

        }

        void reset_state(){
            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    u_hid_f[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    u_hid_s[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c_f[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c_s[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c_f_prev[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    c_s_prev[i][j]=0.0;
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

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    delta_h_f[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    delta_h_s[i][j]=0.0;
                }

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
                    grad_b_h_f[i][j]=0.0;
                }

            for(int i=1;i<=data_num;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_b_h_s[i][j]=0.0;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_h2h_f[i][j]=0.0;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_h2h_s[i][j]=0.0;
                }


            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_s2f[i][j]=0.0;
                }

            for(int i=1;i<=Hidden;i++)
                for(int j=1;j<=Hidden;j++){
                    grad_f2s[i][j]=0.0;
                }

        }


};

int main(){
    MTRNN model;
    Matrix data(20,2);
    Matrix x(1,2);
    Matrix target(1,2);
    Matrix output(1,2);
    Matrix context_f(1,10);
    Matrix context_f_prev(1,10);
    Matrix context_s(1,10);
    Matrix context_s_prev(1,10);
    Matrix c_f_list(data_size,Hidden);
    Matrix c_s_list(data_size,Hidden);
    Matrix z_list(n_time,Output);
    double total_loss = 0;

    std::ifstream fin("lissajous.txt");
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
                    c_f_list[t+1][j] = model.c_f[i][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    c_s_list[t+1][j] = model.c_s[i][j];
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
                    context_f[i][j] = c_f_list[t+1][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    context_f_prev[i][j] = c_f_list[t][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    context_s[i][j] = c_s_list[t+1][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Hidden;j++){
                    context_s_prev[i][j] = c_s_list[t][j];
                }

            for(int i = 1;i<=data_num;i++)
                for(int j = 1;j<=Output;j++){
                    output[i][j] = z_list[t][j];
                }

            model.backward(x,output,context_f,context_f_prev,context_s,context_s_prev,target);

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