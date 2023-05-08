using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Distribution{
    public static float NormalDist(float mu, float sigma){
        //ボックス・ミュラー法による乱数生成
        float alpha = UnityEngine.Random.Range(0.0f,1.0f);
        float beta = UnityEngine.Random.Range(0.0f,1.0f) * Mathf.PI * 2;
        float z = sigma * Mathf.Sqrt(-2 * Mathf.Log(alpha)) * Mathf.Sin(beta) + mu;
        return z;
    }
}