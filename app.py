import os
import numpy as np
import pandas as pd
import pygal
from flask import Flask,render_template,request
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
 
app = Flask(__name__)
df=pd.read_csv(r'C:\Users\new\PycharmProjects\CreditCard\TK12938\creditcard.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
app.config['upload_folder']= r'uploads'
# global df
global path
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load',methods=["POST","GET"])
def load_data():
    if request.method=="POST":
        print('1111')
        files = request.files['file']
        print(files)
        filetype = os.path.splitext(files.filename)[1]
        if filetype == '.csv':
            print('111')
            path = os.path.join(app.config['upload_folder'],files.filename)
            files.save(path)
            print(path)
            return render_template('Load Data.html',msg='valid')
        else:
            return render_template('Load Data.html',msg= 'invalid')
    return render_template('Load Data.html')

@app.route('/preprocess')
def preprocess():
    file = os.listdir(app.config['upload_folder'])
    path =os.path.join(app.config['upload_folder'],file[0])
    df = pd.read_csv(path)
    print(df.head())
    df.isnull().sum()
    return render_template('Pre-process Data.html',msg = 'success')


@app.route('/viewdata',methods=["POST","GET"])
def view_data():
    file = os.listdir(app.config['upload_folder'])
    path = os.path.join(app.config['upload_folder'], file[0])
    df = pd.read_csv(path)
    df1 = df.sample(frac=0.3)
    df1 = df1[:100]
    print(df1)
    return render_template('view data.html',col_name = df1.columns, row_val=list(df1.values.tolist()))




@app.route('/model',methods=["POST","GET"])
def model():
    # global lascore, lpscore, lrscore
    # global nascore, npscore, nrscore
    # global aascore, apscore, arscore
    # global kascore, kpscore, krscore
    global accuracy,recall,precision
    global accuracy1, recall1, precision1
    global accuracy3, recall3, precision3
    global accuracy2, recall2, precision2
    if request.method == "POST":
        model = int(request.form['selected'])
        file = os.listdir(app.config['upload_folder'])
        path = os.path.join(app.config['upload_folder'], file[0])
        df = pd.read_csv(path)
        df1 = df.sample(frac=0.3)
        X = df1.drop(['Time','Class'],axis = 1)
        y= df1.Class
        global train_test_split
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 10)
        if model == 1:
            xgb = XGBClassifier()
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)
            accuracy1 = accuracy_score(y_test,y_pred)
            precision1= precision_score(y_test,y_pred)
            recall1 = recall_score(y_test,y_pred)
            return render_template('model.html',msg='accuracy',score =accuracy1,selected = 'XGBoostClassifier')
        elif model == 2:
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            accuracy2 = accuracy_score(y_test, y_pred)
            precision2 = precision_score(y_test, y_pred)
            recall2 = recall_score(y_test, y_pred)
            return render_template('model.html', msg='accuracy', score=accuracy2, selected='RandomForestClassifier')

        elif model == 3:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            accuracy3 = accuracy_score(y_test, y_pred)
            precision3 = precision_score(y_test, y_pred)
            recall3 = recall_score(y_test, y_pred)
            return render_template('model.html', msg='accuracy', score=accuracy3, selected='DecisionTreeClassifier')
        elif model == 4:
            return render_template('model.html', msg="Please select a model")

    return render_template('model.html')

@app.route('/prediction',methods = ["POST","GET"])
def prediction():

    if request.method=='POST':
        tm = float(request.form['f1'])
        v1 = float(request.form['f2'])
        v2 = float(request.form['f3'])
        v3 = float(request.form['f4'])
        v4 = float(request.form['f5'])
        v5 = float(request.form['f6'])
        v6 = float(request.form['f7'])
        v7 = float(request.form['f8'])
        v8 = float(request.form['f9'])
        v9 = float(request.form['f10'])
        v10 = float(request.form['f11'])
        v11 = float(request.form['f12'])
        v12 = float(request.form['f13'])
        v13 = float(request.form['f14'])
        v14 = float(request.form['f15'])
        v15 = float(request.form['f16'])
        v16 = float(request.form['f17'])
        v17 = float(request.form['f18'])
        v18 = float(request.form['f19'])
        v19 = float(request.form['f20'])
        v20 = float(request.form['f21'])
        v21 = float(request.form['f22'])
        v22 = float(request.form['f23'])
        v23 = float(request.form['f24'])
        v24 = float(request.form['f25'])
        v25 = float(request.form['f26'])
        v26 = float(request.form['f27'])
        v27 = float(request.form['f28'])
        v28 = float(request.form['f29'])
        amt = float(request.form['f30'])

        l = [tm,v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
           v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amt]
        print(l)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        print(x_train.columns)
        print(type(l))
        mna = np.array(l)
        output = clf.predict([mna])
        print(output)
        if output == 0:
            msg = 'Normal'
        else:
            msg = 'Fraud'
        return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')

@app.route('/graph',methods = ["POST","GET"])
def graph():
    print('ihdweud')
    print('jhdbhsgd')
    line_chart = pygal.Bar()
    line_chart.x_labels= ['XGBoostClassifier','RandomForestClassifier','DecisionTreeClassifier']
    print('jdjkfdf')
    line_chart.add('RECALL', [recall1,recall2,recall3])
    print('1')
    line_chart.add('PRECISION', [precision1,precision2,precision3])
    print('2')
    line_chart.add('ACCURACY', [accuracy1,accuracy2,accuracy3])
    print('3')
    graph_data = line_chart.render()
    print('4')
    return render_template('graphs.html', graph_data=graph_data)

if __name__ == ('__main__'):
    app.run(debug=True)
