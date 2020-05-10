#code authored by Akshaj Jain
import streamlit as st
import numpy as np
import pandas as pd
data=np.array([["yes","no","no"],["yes","no","no"],["yes","no","yes"]])
st.title("10-601, Introduction to ML, CMU, Audit by Akshaj, May 2020")
def algo1(data,nf):
    '''data is the data and nf is the number of features, this algo finds the feature which will give the maximum accuracy if independently left'''
    cyes=len(data[data[:,0]=="yes"])
    cno=len(data[data[:,0]=="no"])
    if cyes>cno:
        guess= "yes"
    else:
        guess= "no"
    nf=nf-1
    if cyes==0 or cyes==100 or nf==0:
        return guess
    else:
        score=[0]
        for i in range(1,nf+1):
            score.append(0)
            datano=data[data[:,i]=="no"]
            datayes=data[data[:,i]=="yes"]
            cayesno=len(datano[datano[:,0]=="yes"])
            canono=len(datano[datano[:,0]=="no"])
            if cayesno>canono:
                score[i]=score[i]+cayesno
            else:
                score[i]=score[i]+canono
            cayesyes=len(datayes[datayes[:,0]=="yes"])
            canoyes=len(datayes[datayes[:,0]=="no"])
            if cayesyes>canoyes:
                score[i]=score[i]+cayesyes
            else:
                score[i]=score[i]+canoyes
        f= score.index(max(score))
        return f
        '''datano=data[data[:,f]=="no"]
        datayes=data[data[:,f]=="yes"]'''
opt=st.sidebar.selectbox("Select a Topic",["Decision Trees"])
if opt=="Decision Trees":
    st.write("I read throught the prerequisite available at http://ciml.info/dl/v0_99/ciml-v0_99-ch01.pdf")
    data
    st.write("Here 0th index is the feature that we want to predict")
    st.write("I coded the algorithm to detect the feature while will give us the maximum info gain(using the algorithm one in the given resource)")
    st.write("Maximum info gain in with column")
    st.write(algo1(data,3))
