#code authored by Akshaj Jain
import streamlit as st
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
opag=st.sidebar.selectbox("Topic",["Introduction","Courses/Audits","Entrepreneurship","Schedule","Goals","Equality","AI Policy","Research","Summary of Readings"])
if opag=="Introduction":
    st.markdown("# Interactive blog of my journey at Carnegie Mellon University")
    st.markdown("## About")
    st.write('''I got accepted in Carnegie Mellon University's MS in Artificial Intelligence and Innovation hosted at the LTI, School of Computer Science on the 8th of Feburary 2020. As a 21 year old going into a top AI school, i was interested in changing the world using AI and this degree would allow me to do that. In subsequent months, I decided to make an interative journal/blog about my life at the university. This blog will contain both knowledge and humor and i hope you like it!

NOTE: No code from course assignments/projects has been made public in this blog, since it is against CMU's Academic Integrity Policy, This is a blog, not a place to look for assignment answers.''')
    st.markdown("*Interests at the time of entry:*")
    st.markdown("Entrepreneurship in AI, AI Policy, Reinforcement Learning, Quantum Machine Learning, AI and Ethics")
if opag=="Courses/Audits":
    sesesese=st.sidebar.selectbox("Select Course",["CMU - 10601","CMU - 10725","CMU - 15513"])
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
    if sesesese=="CMU - 10725":
        st.markdown("## Audit of 10-725, Convex Optimization")
    if sesesese=="CMU - 15513":
        opt=st.sidebar.selectbox("Select a Topic",["Everything is Bits","GDB,Assembly and Buffer overflow"])
        st.markdown("## Summary of attendance, 15513, Intorduction to Computer Systems,Summer 2020")
        if opt=="Everything is Bits":
            st.markdown("# Everything is Bits")
            st.markdown('''1. 2's complement representation of integer numbers
2. Floating point representation(IEEE Standard)
3. Casting from one representation to the other one.
4. Little Endian and Big Endian''')
            num=st.slider("Select a number to represent",0,1<<53-1)
            numtohex='{0:014x}'.format(num)
            st.markdown("### Hex representation of the number")
            st.write(numtohex)
            ad=0
            countt=0
            ads=[]
            addys=[]
            while(ad<13):
                ads.append('{0:02x}'.format(countt))
                countt=countt+1
                addy=numtohex[0+ad:ad+2]
                ad=ad+2
                addys.append(addy)
            addyss=addys[::-1]
            addys=np.array(addys).reshape((1,-1))
            addys=pd.DataFrame(addys)
            addys.columns=ads
            st.markdown("### Big Endian Representation")
            st.write(addys)
            addyss=pd.DataFrame(np.array(addyss).reshape((1,-1)))
            addyss.columns=ads
            st.markdown("### Little Endian Representation")
            st.write(addyss)
        if opt=="GDB,Assembly and Buffer overflow":
            st.markdown("# Machine level code")
            st.markdown('''1. How high level code is converted to low level code
2. All Code inevitbly ends up at the low level
3. Registers, storing in registers etc.
4. Procedure Calls
5. Control transfers etc.
6. Buffer overflow and prevention using canaries''')
if opag=="Entrepreneurship":
    st.markdown("# My Entrepreneurial Journey at CMU")
    setime=st.sidebar.selectbox("Time",["May '20","June '20"])
    if setime=="May '20":
        st.markdown('''## May 2020
## Current Goals in Entrepreneurship

1. Open a Startup based on edge analytics

1. Microsoft Imagine Cup

2. Swartz Fellowship

3. CSL Fellowship

4. CSL Course in Spring '21

## Conferences/Events Attended:

1. Talk by [Ed Essey](https://edessey.com/) on Incubation in large corporations
2. Office Hours: J.P Morgan Bankers for Healthcare Startups
3. [Pax Momentum](https://paxmv.com/) Talk by Matthew Hanson
4. CMU Startup Night With OnDeck

## Resources:
1. [How to divide equity in a early stage startup](https://www.cmu.edu/swartz-center-for-entrepreneurship/assets/Connect%20Spring%202017/Frank%20Demmler%20-%20Equity%20Pie/Founders_Pie_Final.pdf)
2. [NVIDIA Startup ecosystem](https://www.nvidia.com/en-us/deep-learning-ai/startups/)''')
    if setime=="June '20":
        st.markdown('''## June 2020
## Current goals
1. Startup in Undisclosed sector
2. Microsoft Imagine Cup(Year long goal)
3. Swartz Fellowship
4. CSL fellowship
5. CSL course in Spring '21

## Conferences Attended:
1. [HBS Virtual Peek Experience](https://www.hbs.edu/mba/admissions/Pages/Virtual-Peek-Experience-2020-abc.aspx)
2. [Critical Considerations for Restarting a More Resilient and Robust US Economy Post COVID-19 Virtual Roundtable by CMU AI](https://www.scs.cmu.edu/calendar/fri-2020-06-12-1030/critical-considerations-restarting-more-resilient-and-robust-us-economy-post-covid-19-virtual-roundtable)

## Talks/Webinars Conducted:
1. Q&A Session with Dr. Hima, Assistant Professor at HBS, For DSCE Students.
2. [Road to NASA by George Salazar](https://www.youtube.com/watch?v=3cqhhpc7eus)''')
if opag=="Schedule":
    pass
if opag=="Goals":
    st.markdown("# Long-Term Goals")
    st.markdown("1. To establish a fair and just society with as few conflicts as possible")
    
if opag=="Equality":
    pass
if opag=="AI Policy":
    st.markdown("## Thoughts and analysis of AI Policy and implementations")
    st.markdown("# [US policy](https://www.whitehouse.gov/ai/)")
    st.markdown(''' Although the US Executive order places alot of importance to AI, and identifies key AI areas it needs to take initiatives in and is well drawn out, the implementation of such a policy seems to be the key problem
                The US policy implementation suggests an increase of 125M$ in R&D of AI/QIC combined which seems  to be alot less than actually required to make a significant impact on such
                areas, nonetheless, the main propogation of US AI continues to be from US companies such as Google, Facebook, Microsoft, Tesla, OpenAI among many others. Alot of startups and privately held research groups have popped up in the field of robotics and AI as well, all of which seem to focus on achieving one of the US AI policy goals.''')
if opag=="Research":
    pass
if opag=="Summary of Readings":
    blahblahbluf=st.sidebar.selectbox("Select book/paper",["Machine Learning by Kevin Murphy"])
    if blahblahbluf=="Machine Learning by Kevin Murphy":
        st.markdown("## Summary of reading of [Machine Learning: A probabilistic approach by Kevin P. Murphy](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)")
        blahbluf=st.sidebar.selectbox("Select Chaper",["Introduction","Probability","Linear Regression"])
        if blahbluf=="Introduction":
            st.markdown("# Chapter 1:Introduction")
            st.markdown('''1. Types of machine learning --> Supervised, Unsupervised and Reinforcement
2. Classification and Regression
3. Clustering
4. Brief of PCA
5. Graph Structures
6. Parametric and Non-Parametric models
7. K-NN algorithm and a probabilistic approach to KNN
8. Curse of dimensionality
9. Liner,Logistic Regression.
10. Overfitting and model selection, K-Fold Cross validation
11. No free Lunch theorem''')
        if blahbluf=="Probability":
            st.markdown("# Chapter 2:Probability")
            st.markdown('''### Frequentist vs Bayesian
Frequentist probability in essence means that the probability implies that a number of events happen, how many of those events will result in a partiucular event. Whereas, in the Bayesian Interpretation, the probability is a way of interpreting the uncertainity surrounding an event.
### Probability Theory
1. The probability of the union of two events is the sum of the probability of the two events from which, the probability of both the events happening is subtracted
2. The probability of both events A and B happening is equivalent to the probability of event A happening, given event B has already happened multiplied by the probability of event B happening
3. The probability of Event A happening given that event B has already happened is equivalent to the probability of both the events happening divided by the probability of event B happening.
4. *Bayes Theorem* : The probability of X happening, given y happens is equivalent to the probability of Y happening given X happens multiplied by the probability of X happening and divided by the summation of probability of y happening given different events in the space of x happen multiplied by the probability of events in the space of x happening.
5. *Unconditional Independence* : Events X&Y are unconditionally independent when the probability of both the events happening is equivalent to the probability of X happening multiplied by the probability of Y happpening, that is, the events' probabilities does not change regardless of the other event happening or not.
6. *Conditional Independence*: Events X&Y are conditionally independent when the probability of both X and Y happening, given another event Z happens is equivalent to the probability of X happening given Z happens multiplied by the probability of Y happening given Z happens
### Continous Random Variables
1. *Cumulative distribution function*: It is a function which contains the probability of all quantities below the given threshold, lets say *x*.
2. Probability of events/values a->b, given $b>a$ is equivalent to $f(b)-f(a)$
3. *Probability Density Function*: $f(x)=\\frac{d}{dx}F(x)$, which essentially gives the probability at a particular point, $x$
4. $P(a<X\\leq b)=\\int_a^b f(x)dx$
5. Quantiles for a continous distribution
6. Expected Values, Variance of a distribution
### Common Discrete distributions
#### *Binomial*
1. Gives the probability of K successes in N number of trials given the probability of success $\\Theta$
1. $\\binom{n}{k}\\Theta^k(1-\\Theta)^{n-k}$ , where, $\\Theta$ is the probability of one success, $n$ is the number of trials and $k$ is the number of successes and $\\binom{n}{k}=\\frac{n!}{(n-k)!k!}$
2. Mean=$\\Theta$
3. Variance = $n\\Theta(1-\\Theta)$
#### *Bernoulli*
1. A Binomial Distrubtion with the event occuring only *once* is known as a Bernoulli Distribution.
2. $\\Theta^x(1-\\Theta)^{1-x}$
#### *Multinomial*
1. Outcomes for events with more than 2 possible outcomes
2. $Mu(x|n,\\Theta)=\\binom{n}{x_{1}...x_{k}}\\prod_{j=1}^{{K}}\\theta_{j}^{x_j}$, where $\\binom{n}{x_{1}...x_{k}}=\\frac{n!}{x_1!x_2!x_3!.....x_K!}$ , $x_j$ is the number of times the event $j$ occcurs
#### *Multinoulli*
1. Outcomes for events with more than 2 possible outcomes but when the event happens only once
2. $Mu(x|1,\\Theta)=\\prod_{j=1}^{k}\\theta_{j}^{x_j}$
#### *Poisson Distribution*
1. $Po(x|\\lambda)=e^{-\\lambda}\\frac{\\lambda^x}{x!}$
2. Applications: Radiactive decay
#### *Empirical Distribution*
1. $p_{emp}(A)=\\frac{1}{N}\\sum_{i=1}^{N}\\delta_{x_i}(A)$ where, $\\delta = \\begin{cases}
   0 &\\text{if } x \\in A  \\\\
   1 &\\text{if } x \\notin A
\\end{cases}$
### Common Continuous distributions
#### Gaussian Distribution
1. Also known as the normal distribution.
2. $p(X=x)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{1}{2\\sigma^2}(x-\\mu)^2}$
3. $\\mu=E [X]$ is the mean and mode.
4. $\\sigma^2=var[X]$
5. $\\sqrt{2\\pi\\sigma^2}$ is the normalization constant to ensure that the density integrates to 1.
6. Precision$(\\lambda)=\\frac{1}{\\sigma^2}$ , a high precision signifies a narrow distribution.
#### Degenerate pdf
1. In a Gaussian, when $\\sigma^2 = 0$, then it is an infinitely tall and thin spike centered at $\\mu$
2. Thus, the probability distribution will be infinity at the mean and 0 everywhere else in such a distribution
#### Student $t$ distribution
1. $T(x|\\mu,\\sigma^2,v) ∝ [1+\\frac{1}{v}(\\frac{x-\\mu}{\\sigma})^2]^{-\\frac{v+1}{2}}$, where $\\mu$ is the mean, $\\sigma^2$ is the scale parameter and the v represents the degrees of freedom
2. mean,mode = $\\mu$, $var = \\frac{v\\sigma^2}{v-2}$
3. Has a heavier tail, allowing for outliers.
#### Laplace distribution
1. ${Lap}(x|\\mu,b)=\\frac{1}{2b}e^{-\\frac{|x-\\mu|}{b}}$
2. mean,mode=$\\mu$, var=$2b^2$
#### Gamma Distribution
1. ${Ga}(T|shape=a,rate=b)= \\frac{b^a}{\\Gamma(a)}T^{a-1}e^{-Tb}$, $x>0, a(shape)>0,b(rate)>0$
2. $\\Gamma(x)=\\int_0^\\infty u^{x-1}e^{-u} du$
3. $mean=\\frac{a}{b}, mode=\\frac{a-1}{b}, var=\\frac{a}{b}^2$
#### Beta Distribution
1. ${Beta}(x|a,b)=\\frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1} , B(a,b)=\\frac{\\Gamma(a)\\Gamma(b)}{\\Gamma(a+b)}
, a,b>0$
2. $mean=\\frac{a}{a+b}, mode=\\frac{a-1}{a+b-2}, var=\\frac{ab}{(a+b)^2(a+b+1)}$
#### Pareto Distribution
1. The Pareto Distribution is used to model heavy tail models.
2. ${Pareto}(x|k,m)=km^kx^{-(k+1)}I(x\\ge m)$
3. $mean=\\frac{km}{k-1} if k\\gt 1, mode=m, var=\\frac{m^2k}{(k-1)^2(k-2) }if k\\gt 2$

''')
        if blahbluf=="Linear Regression":
            st.markdown("# Chapter 7: Linear Regression")
            st.markdown('''''')
            
            
        
