import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import xgboost as xgb
import seaborn as sns

#model รวม


main = st.selectbox('' , ['ข้อมูลทั่วไปของโรคเบาหวาน','กราฟวิเคราะห์ข้อมูล','เปรียบเทียบโมเดล','ลองทำนายความเสี่ยงกัน'])
if main == 'ข้อมูลทั่วไปของโรคเบาหวาน' : 

    choose = st.sidebar.selectbox('กรุณาเลือกหัวข้อที่สนใจ' , ['โรคเบาหวานคืออะไร','ประเภทของโรคเบาหวาน','สาเหตุของโรคเบาหวาน','อาการของโรคเบาหวาน','โรคแทรกซ้อน','ห่างไกลโรคเบาหวาน'])
    if choose == 'โรคเบาหวานคืออะไร' :
        from PIL import Image
        img = Image.open('p01.png') 
        st.image(img,caption='ดูข้อมูลเพิ่มเติม คลิ๊ก! ลูกศรซ้ายมือ')

        from PIL import Image
        img = Image.open('p05.png') 
        st.image(img,width=550)


    

    if choose == 'ประเภทของโรคเบาหวาน' :
        from PIL import Image
        img = Image.open('p03.png') 
        st.image(img,width=680)

    if choose == 'สาเหตุของโรคเบาหวาน' :
        from PIL import Image
        img = Image.open('p06.png') 
        st.image(img,width=640)

    if choose == 'อาการของโรคเบาหวาน' :
        from PIL import Image
        img = Image.open('p07.png') 
        st.image(img,width=640)

    if choose == 'โรคแทรกซ้อน' :
        from PIL import Image
        img = Image.open('p08.png') 
        st.image(img,width=640)

    if choose == 'ห่างไกลโรคเบาหวาน' :
        from PIL import Image
        img = Image.open('p09.png') 
        st.image(img,width=640)


#หน้า2 

if main == 'กราฟวิเคราะห์ข้อมูล' :
    from PIL import Image
    img = Image.open('gg1.png')
    st.image(img)


    #load data 
    data = pd.read_csv('diabetes.csv')

    #visualize data

    #columnzone5
    col13,col14 = st.beta_columns(2)
    with col14 :
        g1 = sns.relplot(x='SkinThickness', y='Insulin',hue='Outcome',sizes=(40, 400), palette="muted",height=5, data=data)
        st.pyplot(g1)
    with col13 :
        from PIL import Image
        img = Image.open('g16.png')
        st.image(img)


    col1,col2 = st.beta_columns(2)
    with col1 : 
        g1 = sns.relplot(x='Glucose', y='Insulin',hue='Outcome',sizes=(40, 400), palette="muted",height=5, data=data)
        st.pyplot(g1)
    with col2 :
        from PIL import Image
        img = Image.open('g1pic.png')
        st.image(img)


    col3,col4 = st.beta_columns(2)
    with col3 :
        from PIL import Image
        img = Image.open('g2pic.png')
        st.image(img)
    with col4 :
        g2 = sns.relplot(x="Glucose",y="BMI",hue='Outcome',sizes=(40, 400), palette="muted",height=5, data=data)
        st.pyplot(g2)

    col5,col6=st.beta_columns(2)
    with col5 :
        g3 = sns.relplot(y="SkinThickness",x="BMI",hue='Outcome',sizes=(40, 400), palette="muted",height=5, data=data)
        st.pyplot(g3)
    with col6 :
        from PIL import Image
        img = Image.open('g110.png')
        st.image(img)


if main == 'เปรียบเทียบโมเดล' :

    #load data 
    data = pd.read_csv('diabetes.csv')

    #preprocessing data 
    data['Age'] = pd.qcut(data['Age'], 10, duplicates='drop')
    data['BMI'] = pd.qcut(data['BMI'], 5, duplicates='drop')
    data = pd.get_dummies(data)
    X = data.drop(columns=['Outcome']).values
    y = data['Outcome'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    fill_values  = SimpleImputer(missing_values=0, strategy='mean')#แทน 0 ด้วยค่าเฉลี่ย mean
    X_train = fill_values.fit_transform(X_train)
    X_test  = fill_values.fit_transform(X_test)

    #function trainmodel
    def model(model) :
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc = accuracy_score(predict,y_test)
        return acc

    #เปรียบเทียบโมเดล


    from PIL import Image
    img = Image.open('g111.png')
    st.image(img)


    #model
    col1,col2,col3 = st.beta_columns(3)
    with col1 :
        st.write('KNN')
        clfK = KNeighborsClassifier(n_neighbors = 9)
        accK = model(clfK)
        st.write('acc : ' , accK)

        st.write('SVC')
        clfS=SVC(C=1,gamma =0.001)
        accS = model(clfS)
        st.write('acc : ' , accS)

    with col2 :
        st.write('Desicion tree')
        clfD=tree.DecisionTreeClassifier(max_depth=2)
        accD = model(clfD)
        st.write('acc : ' , accD)

        st.write('Random tree')
        clfR=RandomForestClassifier(max_depth=10,random_state=10)
        accR = model(clfR)
        st.write('acc : ' , accR)

    with col3 :
        st.write('XgBoost')
        clfX=xgb.XGBClassifier(max_depth=6)
        accX = model(clfX)
        st.write('acc : ' , accX)

        st.write('LogisticRegression')
        clfL = LogisticRegression(C=5)
        accL = model(clfL)
        st.write('acc : ' , accL)

    #plotgraph


    st.write('\n')
    st.write('\n')
    model = ['Logistic Regression', 'K Nearest Neighbors', 'Random Forests', 'Support Vector Machines',
         'XGBoost', 'Desion tree']
    score = [accL, accK, accR, accS , accX, accD]

    plt.figure(figsize = (12,6))
    sns.barplot(x = model, y = score, palette = 'magma')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False) #ซ่อนคำเตือน

if main == 'ลองทำนายความเสี่ยงกัน' :
    #choose data 
    def user_feature() :
        from PIL import Image
        img = Image.open('p10.png') 
        st.image(img)
        col1,col2,col3,col4 = st.beta_columns(4)
        with col1 :
            pregnant = st.slider('จำนวนการตั้งครรภ์',0,20,0)
            glucose  = st.slider('ระดับกลูโคส(2ชั่วโมง)',0,250,0)
        with col2 :
            blood    = st.slider('แรงดันเลือด(mm Hg)',0,200,0)
            skin     = st.slider('ความหนาของผิวหนัง(mm)',0,150,0)
        with col3 :
            insulin  = st.slider('ระดับอินซูลีน(mu U/ml)',0,800,0)
            bmi      = st.slider('BMI',0.0,80.0,00.0)
        with col4 :
            dpf      = st.slider('DPF',0.000,3.000,0.000)
            age      = st.slider('อายุ(ปี)',0,100,0)

        data = {'การตั้งครรภ์' : pregnant , 
                'กลูโคส' : glucose ,
                'แรงดันเลือด' : blood , 
                'ผิวหนัง' : skin ,
                'อินซูลีน' : insulin,
                'BMI' : bmi,
                'DPF' : dpf,
                'อายุ' : age}
        features = pd.DataFrame(data,index=[0])
        return features

    features = user_feature()
    st.write('---')
    st.write(features)
    st.write('-----')

# Load pima dataset
    def load_data() :
        data = pd.read_csv('pima-data.csv')
        X = data.drop(columns=['diabetes']).values
        y = data['diabetes'].values
        return X,y

    #preprocessing data
    X,y = load_data()
    model = RandomForestClassifier(max_depth=10,random_state=10)
 
    #classification  train model

    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.2, random_state=1234)
    model.fit(X_train,y_train)
    from sklearn.impute import SimpleImputer 
    fill_values  = SimpleImputer(missing_values=0, strategy='mean')

#แทน 0 ด้วยค่าเฉลี่ย mean

    X_train = fill_values.fit_transform(X_train)
    X_test  = fill_values.fit_transform(X_test)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    #predict data ที่รับมา
    predict = model.predict(features)

    col1,col2,col3 = st.beta_columns(3)
    with col2 :
        st.write('มาดูผลคำทำนายกันเถอะทุกคน')
        if predict[0] == 0 :
            from PIL import Image
            img = Image.open('smile.jpg') 
            st.image(img,width=200,caption='เย่! คุณมีความเสี่ยงค่อนข้างน้อย')
        else :
            from PIL import Image
            img = Image.open('bad.jpg') 
            st.image(img,width=200,caption='แง! คุณมีความเสี่ยงค่อนข้างมาก')
    

