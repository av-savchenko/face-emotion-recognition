package com.asav.facialprocessing;

import java.io.Serializable;


/**
 * Created by avsavchenko.
 */
public class FaceData implements ClassifierResult,Serializable {
    public double age=0;
    public float maleScore = 0; //male probability
    public float[] ethnicityScores=null;
    public float[] features=null;

    public FaceData(){

    }
    public FaceData(double age, float maleScore, float[] ethnicityScores, float[] features){
        this.age=age;
        this.maleScore =maleScore;

        if (ethnicityScores!=null) {
            this.ethnicityScores = new float[ethnicityScores.length];
            System.arraycopy(ethnicityScores, 0, this.ethnicityScores, 0, ethnicityScores.length);
        }
        else{
            this.ethnicityScores =null;
        }

        this.features=new float[features.length];
        System.arraycopy( features, 0, this.features, 0, features.length );
    }

    public int getAge(){
        return (int) Math.round(age);
    }
    public boolean isMale(){
        return isMale(maleScore);
    }
    public static boolean isMale(double maleScore){
        return maleScore >= 0.6;
    }

    private static String[] ethnicities={"","white", "black", "asian", "indian", "latino/middle Eastern"};
    public static String getEthnicity(float[] ethnicityScores){
        int bestInd=-1;
        if (ethnicityScores!=null){
            float maxScore=0;
            for(int i=0;i<ethnicityScores.length;++i){
                if(maxScore<ethnicityScores[i]){
                    maxScore=ethnicityScores[i];
                    bestInd=i;
                }
            }
        }
        return ethnicities[bestInd+1];
    }
    public String toString(){
        return String.format("age=%d %s %s",getAge(),isMale()? " male" : " female",getEthnicity(ethnicityScores));
    }
}
