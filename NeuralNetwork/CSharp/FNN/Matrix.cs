using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static Distribution;

public class Matrix
{
    private int rows; //行数
    private int columns; //列数
    private float[,] elements; //要素
    public int Rows
    {
        get{return rows;}
    }

    public int Columns
    {
        get{return columns;}
    }

    public float[,] Elements
    {
        get{return elements;}
    }

    //コンストラクタ
    public Matrix(int row, int column)
    {
        rows = row;
        columns = column;
        elements = new float[rows,columns];
    }

    //行列の加算
    public static Matrix operator +(Matrix a,Matrix b)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = a.elements[i, j] + b.elements[i, j];
                }
            }
        return result;
    }

    //行列と定数の加算(前)
    public static Matrix operator +(float b,Matrix a)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = b + a.elements[i, j];
                }
            }
        return result;
    }

    //行列と定数の加算(後)
    public static Matrix operator +(Matrix a, float b)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = a.elements[i, j] + b;
                }
            }
        return result;
    }

    //行列の減算
    public static Matrix operator -(Matrix a,Matrix b)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = a.elements[i, j] - b.elements[i, j];
                }
            }
        return result;
    }

    //行列と定数の減算(前)
    public static Matrix operator -(float b,Matrix a)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = b - a.elements[i, j];
                }
            }
        return result;
    }

    //行列と定数の減算(後)
    public static Matrix operator -(Matrix a, float b)
    {
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = a.elements[i, j] - b;
                }
            }
        return result;
    }

    //行列の乗算
    public static Matrix operator *(Matrix a, Matrix b)
    {
        //アダマール積(要素積)の場合
        if((a.rows == b.rows) && (a.columns == b.columns))
        {
            Matrix result = new Matrix(a.rows,a.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = a.elements[i, j] * b.elements[i, j];
                }
            }
            return result;
        //行列積の場合
        }else if(a.columns == b.rows){
            Matrix result = new Matrix(a.rows,b.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < b.columns; j++)
                {
                    for (int k = 0; k < a.columns; k++)
                    {
                        result.elements[i, j] += a.elements[i, k] * b.elements[k, j];
                    }
                }
            }
            return result;
        }else{
            Debug.LogError("cannot product");
            return null;
        }
    }

    //行列の定数倍(前)
    public static Matrix operator *(float b, Matrix a)
    {
        Matrix result = new Matrix(a.rows,a.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = b * a.elements[i, j];
                }
            }
            return result;
    }

    //行列の定数倍(後)
    public static Matrix operator *(Matrix a, float b)
    {
        Matrix result = new Matrix(a.rows,a.columns);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.columns; j++)
                {
                    result.elements[i, j] = b * a.elements[i, j];
                }
            }
            return result;
    }

    //0で初期化
    public void Zero()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                elements[i, j] = 0.0f;
            }
        }
    }

    //1で初期化
    public void One()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                elements[i, j] = 1.0f;
            }
        }
    }

    /// <summary>
    /// 正規分布による初期化
    /// </summary>
    /// <param name="mu">平均</param>
    /// <param name="sigma">分散</param>
    public void Normal(float mu, float sigma)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                elements[i, j] = NormalDist(mu,sigma);
            }
        }
    }

    //転置
    public Matrix transpose()
    {
        //行と列の数が反転する
        int t_rows = columns;
        int t_columns = rows;
        Matrix result = new Matrix(t_rows,t_columns);
        for (int i = 0; i < t_rows; i++)
        {
            for (int j = 0; j < t_columns; j++)
            {
                result.elements[i, j] = elements[j, i];
            }
        }
        return result;
    }

    //二乗
    public Matrix Power()
    {
        Matrix result = new Matrix(rows,columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result.elements[i, j] = elements[i,j] * elements[i,j];
            }
        }
        return result;
    }

    //表示用
    public void Show(string name="matrix")
    {
        int i = 0;
        string str = name + "\n";

        foreach(var element in elements){
            i += 1;
            str += element;
            if((i % columns) == 0){
                str += "\n";
            }else{
                str += " ";
            }
        }
        Debug.Log(str);
    }

    /// <summary>
    /// 行列保存用メソッド
    /// </summary>
    /// <param name="name">ファイル名</param>
    /// <param name="folderName">フォルダ名</param>
    /// <param name="isNewFile">連番ファイル名を作るかどうか</param>
    public void Save(string name="MatrixData", string folderName="MatrixSaveData", bool isNewFile=false)
    {
        int number = 1;
        string folderPath = Application.dataPath + "/" + folderName + "/";
        if(!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }
        string filenameTemp = folderPath + name;
        string filename = filenameTemp; 
        if(isNewFile)
        {
            while(File.Exists(filename + ".txt"))
            {
                filename = $"{filenameTemp}_{++number}";
            }
        }

        int i = 0;
        StreamWriter streamWriter = new StreamWriter(filename + ".txt");
        foreach(var element in elements){
            i += 1;
            if((i % columns) == 0){
                streamWriter.Write(element + "\n");
            }else{
                streamWriter.Write(element + ",");
            }
        }
        streamWriter.Flush();
        streamWriter.Close();
    }

    public void Load(string name="MatrixData", string folderName="MatrixSaveData")
    {
        string filename = Application.dataPath + "/" + folderName + "/" + name + ".txt";
        StreamReader streamReader = new StreamReader(filename);
        string strStream = streamReader.ReadToEnd();
        System.StringSplitOptions option = StringSplitOptions.RemoveEmptyEntries;
        string []lines = strStream.Split(new char[]{'\r','\n'},option);
        char []spliter = new char[1]{','};
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                //カンマ分け
                string [ ] readStrData = lines[i].Split(spliter, option);
                //型変換
                elements[i, j] = float.Parse(readStrData[j]);
                
            }
        }
	streamReader.Close();

    }
}