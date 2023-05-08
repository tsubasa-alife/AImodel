using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Function;

public class FNN
{
    public Dictionary<string,Matrix> Params = new Dictionary<string,Matrix>();
    public Dictionary<string,Matrix> Grads = new Dictionary<string,Matrix>();
    public Dictionary<string,Matrix> Deltas = new Dictionary<string,Matrix>();
    public Dictionary<string,Matrix> State = new Dictionary<string,Matrix>();

    public FNN(int InputNum, int HiddenNum, int OutputNum, int Dimension=1)
    {
        //重みとバイアスの宣言
        Params.Add("i2h",new Matrix(InputNum,HiddenNum));
        Params.Add("c2o",new Matrix(HiddenNum,OutputNum));
        Params.Add("b_h",new Matrix(Dimension,HiddenNum));
        Params.Add("b_o",new Matrix(Dimension,OutputNum));
        //勾配の宣言
        Grads.Add("i2h",new Matrix(InputNum,HiddenNum));
        Grads.Add("c2o",new Matrix(HiddenNum,OutputNum));
        Grads.Add("b_h",new Matrix(Dimension,HiddenNum));
        Grads.Add("b_o",new Matrix(Dimension,OutputNum));
        //内部状態の宣言
        State.Add("input",new Matrix(Dimension,InputNum));
        State.Add("hidden",new Matrix(Dimension,HiddenNum));
        State.Add("context",new Matrix(Dimension,HiddenNum));
        State.Add("output",new Matrix(Dimension,OutputNum));
        //デルタ式の宣言
        Deltas.Add("hidden",new Matrix(Dimension,HiddenNum));
        Deltas.Add("context",new Matrix(Dimension,HiddenNum));
        Deltas.Add("output",new Matrix(Dimension,OutputNum));
        //正規分布によるパラメタの初期化
        List<string> keylist = new List<string>(Params.Keys);
        foreach(var key in keylist)
        {
            Params[key].Normal(0f,1f);
        }
    }

    //順方向計算
    public void forward(Matrix input)
    {
        State["input"] = input;
        State["hidden"] = State["input"] * Params["i2h"] + Params["b_h"];
        State["context"] = tanh(State["context"]);
        State["output"] = tanh(State["context"] * Params["c2o"] + Params["b_o"]);
    }

    //逆方向計算
    public void backward(Matrix target)
    {
        Deltas["output"] = (State["output"] - target) * (1.0f - State["output"].Power());

        Grads["c2o"] += State["context"].transpose() * Deltas["output"];
        Grads["b_o"] += Deltas["output"];

        Deltas["context"] = Deltas["output"] * Params["c2o"].transpose();
        Deltas["hidden"] = Deltas["context"] * (1.0f - State["context"].Power());

        Grads["i2h"] += State["input"].transpose() * Deltas["hidden"];
        Grads["b_h"] += Deltas["hidden"];
    }

    //パラメタの更新(SGD)
    public void SGD(float eta=0.01f)
    {
        List<string> keylist = new List<string>(Params.Keys);
        foreach(var key in keylist)
        {
            Params[key] = Params[key] - eta * Grads[key];
        }
    }

    //内部状態の初期化
    public void reset_state()
    {
        List<string> keylist = new List<string>(State.Keys);
        foreach(var key in keylist)
        {
            State[key].Zero();
        }
    }

    //デルタ式の初期化
    public void clear_deltas()
    {
        List<string> keylist = new List<string>(Deltas.Keys);
        foreach(var key in keylist)
        {
            Deltas[key].Zero();
        }
    }

    //勾配の初期化
    public void clear_grads()
    {
        List<string> keylist = new List<string>(Grads.Keys);
        foreach(var key in keylist)
        {
            Grads[key].Zero();
        }
    }

    //パラメタの表示
    public void ShowParams()
    {
        List<string> keylist = new List<string>(Params.Keys);
        foreach(var key in keylist)
        {
            Params[key].Show(key);
        }
    }

}
