using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Function
{
    //活性化関数tanh
    public static Matrix tanh(Matrix a)
    {
        Matrix result = new Matrix(a.Rows,a.Columns);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result.Elements[i, j] = (Mathf.Exp(a.Elements[i,j]) - Mathf.Exp(-a.Elements[i,j])) / (Mathf.Exp(a.Elements[i,j]) + Mathf.Exp(-a.Elements[i,j]));
            }
        }

        return result;
    }
}