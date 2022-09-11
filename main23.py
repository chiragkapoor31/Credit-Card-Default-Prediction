from flask import *
from main import predict_default
app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return render_template("success.html",data=[{'gender': 'Female'}, {'gender': 'Male'}] , data1=[{'Maritial_Status': 'Single'}, {'Maritial_Status': 'Married'}],data2=[{'grad_school': 'No'}, {'grad_school': 'Yes'}],data3=[{'university': 'No'}, {'university': 'Yes'}],data4=[{'high_school': 'No'}, {'high_school': 'Yes'}] ,name=f.filename)


@app.route('/output' , methods = ['POST'])
def output():
    limit_bal = request.form['limit_bal']

    age       = request.form['age']

    pay_0 = request.form['pay_0']
    pay_2 = request.form['pay_2']
    pay_3 = request.form['pay_3']
    pay_4 = request.form['pay_4']
    pay_5 = request.form['pay_5']
    pay_6 = request.form['pay_6']

    pay_amt1 = request.form['pay_amt1']
    pay_amt2 = request.form['pay_amt2']
    pay_amt3 = request.form['pay_amt3']
    pay_amt4 = request.form['pay_amt4']
    pay_amt5 = request.form['pay_amt5']
    pay_amt6 = request.form['pay_amt6']

    bill_amt1 = request.form['bill_amt1']
    bill_amt2 = request.form['bill_amt2']
    bill_amt3 = request.form['bill_amt3']
    bill_amt4 = request.form['bill_amt4']
    bill_amt5 = request.form['bill_amt5']
    bill_amt6 = request.form['bill_amt6']

    grad_school = request.form.get('comp_select2')
    if grad_school == 'No':
        grad_school = 0
    else:
        grad_school = 1

    university = request.form.get('comp_select3')
    if university == 'No':
        university = 0
    else:
        university = 1

    high_school = request.form.get('comp_select4')
    if high_school == 'No':
        high_school = 0
    else:
        high_school = 1

    Gender = request.form.get('comp_select')
    if Gender == 'Male':
        Gender = 1
    else:
        Gender = 0

    Maritial_Status = request.form.get('comp_select1')
    if Maritial_Status == 'Single':
        Maritial_Status = 0
    else:
        Maritial_Status = 1

    a = predict_default(limit_bal, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt1, bill_amt2, bill_amt3,
                        bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
                        grad_school, university, high_school, Gender, Maritial_Status)
    if a == [0]:
        prediction = "Will Not Default"
    else:
        prediction = "Will Default"
    

    return render_template("output.html",limit_bal = limit_bal , age = age , pay_0 = pay_0 , pay_2 = pay_2 , pay_3 = pay_3 , pay_4 = pay_4 , pay_5 = pay_5 , pay_6=pay_6 , bill_amt1 = bill_amt1 , bill_amt2 = bill_amt2 , bill_amt3 = bill_amt3 , bill_amt4 = bill_amt4 , bill_amt5 = bill_amt5 , bill_amt6 = bill_amt6 , pay_amt1 = pay_amt1 , pay_amt2 = pay_amt2 , pay_amt3 = pay_amt3 , pay_amt4 = pay_amt4 , pay_amt5 = pay_amt5 , pay_amt6 = pay_amt6 , grad_school = grad_school , university = university  , high_school = high_school , Gender =Gender , Maritial_Status = Maritial_Status , prediction =prediction )



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=50001, debug=True)
