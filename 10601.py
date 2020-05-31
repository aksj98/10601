#code authored by Akshaj Jain
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
opag=st.sidebar.selectbox("Topic",["Introduction","Courses/Audits","Entrepreneurship","Schedule","Goals","Equality","AI Policy"])
if opag=="Introduction":
    st.markdown("# Interactive blog of my journey at Carnegie Mellon University")
    st.markdown("## About")
    st.write('''I got accepted in Carnegie Mellon University's MS in Artificial Intelligence and Innovation hosted at the LTI, School of Computer Science on the 8th of Feburary 2020. As a 21 year old going into a top AI school, i was interested in changing the world using AI and this degree would allow me to do that. In subsequent months, I decided to make an interative journal/blog about my life at the university. This blog will contain both knowledge and humor and i hope you like it!

NOTE: No code from course assignments/projects has been made public in this blog, since it is against CMU's Academic Integrity Policy, This is a blog, not a place to look for assignment answers.''')
    st.markdown("*Interests at the time of entry:*")
    st.markdown("Entrepreneurship in AI, AI Policy, Reinforcement Learning, Quantum Machine Learning, AI and Ethics")
if opag=="Courses/Audits":
    sesesese=st.sidebar.selectbox("Select Course",["CMU - 10601","CMU - 10725","CMU-15513"])
    if sesesese=="CMU - 10601":
        st.markdown('''## Audit of 10-601, Introduction to ML
[Official Course Website](http://www.cs.cmu.edu/~mgormley/courses/10601/index.html)''')
        def bayesoptimalclassifier(x):
            return max_probability(x,y)
        def spinaprox():
            a=np.arange(0,2*3.14,2*3.14/100)
            b=np.sin(a)
            plt.plot(a,b)
            plt.plot(a,[0]*100)
            plt.title("sine wave approximation")
            return a,b
        def aprox(n,a,b):
            c=np.random.randint(0,100,size=n)
            c.sort()
            aproxidataa=a[c]
            aproxidatab=b[c]
            plt.plot(aproxidataa,aproxidatab)
        def sqloss(y,yhat):
            return (y-yhat)**2
        def binloss(y,yhat):
            if y==yhat:
                return 0
            else:
                return 1
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
        opt=st.sidebar.selectbox("Select a Topic",["Decision Trees","K-Nearest Neighbour","Perceptron"])
        if opt=="Decision Trees":
            st.markdown(" ## Decision Trees - Lecture 1")
            data=np.array([["yes","no","no"],["yes","no","no"],["yes","no","yes"]])
            st.markdown("Reading-1 available at [Chapter 1,Course in ML](http://ciml.info/dl/v0_99/ciml-v0_99-ch01.pdf) by [Dr. Hal](http://users.umiacs.umd.edu/~hal/index.html)")
            data
            st.write("Here 0th index is the feature that we want to predict")
            st.write("I coded the algorithm to detect the feature while will give us the maximum info gain(using the algorithm one in the given resource)")
            st.write("Maximum info gain in with column")
            st.write(algo1(data,3))
            #add sin curve with approximation using n calls
            aabcd,bacd=spinaprox()
            numpoints=st.slider('Number of points to approx with')
            aprox(n=numpoints,a=aabcd,b=bacd)
            st.pyplot()
            st.markdown("## Decision Trees - Lecture 2")
            st.markdown("Reading-2 available at [Chapter 2,Course in ML](http://ciml.info/dl/v0_99/ciml-v0_99-ch02.pdf) by [Dr. Hal](http://users.umiacs.umd.edu/~hal/index.html)")
            def train(data):
                #root=new Node(data=data)
                return train_tree(root)
        if opt=="K-Nearest Neighbour":
            def knnpre(d,k,x):
                ds=(np.array(d))[:,1:]
                di=np.sum(((ds-x)**2),axis=1)
                di=di.reshape((20,1))
                rat=np.array(d)[:,0]
                rat=rat.reshape((20,1))
                final=np.concatenate((di,rat),axis=1)
                final=final[final[:,0].argsort()]
                yeze=0
                for i in range(0,k):
                    if (final[i,1]>-1):
                        yeze=yeze+1
                    else:
                        yeze=yeze-1
                if yeze>-1:
                    return 1
                else:
                    return 0
            st.markdown('''## K-Nearest Neighbour - Lecture 3
Reading - 1 available at [Chapter 3, Geometry and Nearest Neighbors](http://ciml.info/dl/v0_99/ciml-v0_99-ch03.pdf) by [Dr. Hal](http://users.umiacs.umd.edu/~hal/index.html)''')
            tabl1=pd.DataFrame({"Rating":[2,2,2,2,2,1,1,1,0,0,0,0,-1,-1,-1,-1,-2,-2,-2,-2],"Ez":[1,1,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,1],"AI":[1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0,0],"Sys":[0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],"Th":[1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,0,0,0],"Morn":[0,0,0,0,1,0,0,0,1,1,0,1,1,0,1,1,0,1,0,1]},columns=["Rating","Ez","AI","Sys","Th","Morn"])
            '''Course Rating Dataset'''
            tabl1
            X=tabl1[tabl1['Rating']<0]
            Xp=tabl1[tabl1['Rating']>=0]
            xaxis=st.selectbox("X-Axis",["AI","Ez","Sys"])
            yaxis=st.selectbox("Y-Axis",["AI","Ez","Sys"])
            plt.scatter(X[xaxis],X[yaxis],c='r')
            plt.scatter(Xp[xaxis],Xp[yaxis],marker='+',c='b')
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            plt.title("Plotting the dataset")
            st.pyplot()
            st.markdown("## K-NN on the Dataset")
            slidee=st.slider("Select the element to predict",1,20)
            st.markdown(" Element is")
            cur_element=tabl1.loc[slidee-1,:]
            st.write(cur_element)
            slidee2=st.slider("Select the K",1,20)
            reseult=knnpre(tabl1,slidee2,np.array(tabl1)[slidee-1,1:])
            if (reseult)==0:
                st.markdown('''Prediction:
*Rating is Negative*''')
            else:
                st.markdown('''Prediction:
*Rating is Positive*''')
            if (( (reseult==1) and (cur_element['Rating']>-1) ) or ((reseult==0) and (cur_element['Rating']<0))):
                st.markdown("#### Prediction is correct")
            else:
                st.markdown("#### Prediction is incorrect")
            #bayes optimal classifier
            #inductive bias: how much does the model prefer a solution
            #sources of error: 1) Noise in training data 2) Noise in Feature or label 3) limited features 4) Misaligned bias
        if opt=="Perceptron":
            st.markdown('''## The Perceptron
Reading available at [Perceptron](http://ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf) by [Dr. Hal](http://users.umiacs.umd.edu/~hal/index.html)''')
            def perceptrain(d,epocs,y):
                w=np.zeros(d.shape)
                b=np.array([0])
                for i in range(epocs):
                    a=w[i,:]*d[i,:]+b
                    if y<=0:
                        w[i,:]
    if sesesese=="10725":
        st.markdown("## Audit of 10-725, Convex Optimization")
    if sesesese=="CMU-15513":
        st.markdown("## Summary of attendance, 15513, Intorduction to Computer Systems,Summer 2020")
        st.markdown("# Topic 1")
        st.markdown('''1. 2's complement representation of integer numbers
2. Floating point representation(IEEE Standard)
3. Casting from one representation to the other one.
4. Little Endian and Big Endian''')
if opag=="Entrepreneurship":
    st.markdown("# My Entrepreneurial Journey at CMU")
    setime=st.sidebar.selectbox("Time",["May '20"])
    if setime=="May '20":
        st.markdown('''## May 2020
## Current Goals in Entrepreneurship

1. Open a Startup based on edge analytics

1. MiT Solve

2. Swartz Fellowship

3. CSL Fellowship

4. CSL Course in Spring '21

## Conferences/Events Attended:

1. Talk by [Ed Essey](https://edessey.com/) on Incubation in large corporations
2. Office Hours: J.P Morgan Bankers for Healthcare Startups
3. [Pax Momentum](https://paxmv.com/) Talk by Matthew Hanson
4. CMU Startup Night With OnDeck

## Resources:
1. [How to divide equity in a early stage startup](https://www.cmu.edu/swartz-center-for-entrepreneurship/assets/Connect%20Spring%202017/Frank%20Demmler%20-%20Equity%20Pie/Founders_Pie_Final.pdf)''')
if opag=="Schedule":
    pass
if opag=="Goals":
    pass
if opag=="Equality":
    pass
if opag=="AI Policy":
    st.markdown("## Thoughts and analysis of AI Policy and implementations")
    st.markdown("# [US policy](https://www.whitehouse.gov/ai/)")
    st.markdown(''' Although the US Executive order places alot of importance to AI, and identifies key AI areas it needs to take initiatives in and is well drawn out, the implementation of such a policy seems to be the key problem
                The US policy implementation suggests an increase of 125M$ in R&D of AI/QIC combined which seems  to be alot less than actually required to make a significant impact on such
                areas, nonetheless, the main propogation of US AI continues to be from US companies such as Google, Facebook, Microsoft, Tesla, OpenAI among many others. Alot of startups and privately held research groups have popped up in the field of robotics and AI as well, all of which seem to focus on achieving one of the US AI policy goals.''')
    
    
