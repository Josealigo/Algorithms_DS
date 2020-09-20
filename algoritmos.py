import pandas as pd
import numpy as np
import re
import math
import sympy

e = np.e
sin = math.sin
cos = math.cos
tan = math.tan

#Evaluación REGREX
def evaluate_Fx(str_equ, valX):
    x = float(valX)
    strOut = str_equ
    strOut = strOut.replace("^", "**")
    exp_chage = [a for a in re.findall(r"[0-9]x", strOut )]
    for i in exp_chage:
        num = i.replace("x","")
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]\(", strOut )]
    for i in exp_chage:
        num = i.replace("(","")
        new_exp = num + '*('
        strOut = strOut.replace(i,new_exp)
    out = eval(strOut)
    return out

def evaluate_Fxy(str_equ, valX, valY):
    x = float(valX)
    y = float(valY)
    e = np.e
    sin = math.sin
    cos = math.cos
    tan = math.tan
    strOut = str_equ
    strOut = strOut.replace("^", "**")
    exp_chage = [a for a in re.findall(r"[0-9]x", strOut )]
    for i in exp_chage:
        num = i.replace("x","") 
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]y", strOut )]
    for i in exp_chage:
        num = i.replace("y","") 
        new_exp = num + '*y'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"xy", strOut )]
    for i in exp_chage:
        num = i.replace("y","") 
        new_exp = num + '*y'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"yx", strOut )]
    for i in exp_chage:
        num = i.replace("x","") 
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]\(", strOut )]
    for i in exp_chage:
        num = i.replace("(","")
        new_exp = num + '*('
        strOut = strOut.replace(i,new_exp)
    out = eval(strOut)
    return out


#print(evaluate_Fxy("2xy^3 - 3y^2 + xy + 1", 1, 1)) #expected value is 1




def real_derivative_Fx(str_equ):

    strOut = str_equ
    strOut = strOut.replace("^", "**")
    exp_chage = [a for a in re.findall(r"[0-9]x", strOut )]
    for i in exp_chage:
        num = i.replace("x","")
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]\(", strOut )]
    for i in exp_chage:
        num = i.replace("(","")
        new_exp = num + '*('
        strOut = strOut.replace(i,new_exp)
    x = sympy.Symbol('x')
    y = eval(strOut)
    yprime = y.diff(x)
    
    return str(yprime)

def real_derivative_Fxy(str_equ):

    strOut = str_equ
    strOut = strOut.replace("^", "**")
    exp_chage = [a for a in re.findall(r"[0-9]x", strOut )]
    for i in exp_chage:
        num = i.replace("x","") 
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]y", strOut )]
    for i in exp_chage:
        num = i.replace("y","") 
        new_exp = num + '*y'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"xy", strOut )]
    for i in exp_chage:
        num = i.replace("y","") 
        new_exp = num + '*y'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"yx", strOut )]
    for i in exp_chage:
        num = i.replace("x","") 
        new_exp = num + '*x'
        strOut = strOut.replace(i,new_exp)
    exp_chage = [a for a in re.findall(r"[0-9]\(", strOut )]
    for i in exp_chage:
        num = i.replace("(","")
        new_exp = num + '*('
        strOut = strOut.replace(i,new_exp)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    fxy = eval(strOut)
    dx_df = fxy.diff(x)
    dy_df = fxy.diff(y)
    
    return str(dx_df), str(dy_df)

#print(real_derivative_Fxy("2xy^3 - 3y^2 + xy + 1"))
#Diferencias finitas centradas

def finite_centered_differences(str_equ, x, h):
    h = float(h)
    positive_x = float(x) + h
    negative_x = float(x) - h
    out = (evaluate_Fx(str_equ,positive_x) - evaluate_Fx(str_equ,negative_x)) /(2*h)
    return out

def finite_centered_differences_xy(str_equ, x, y, h):
    h = float(h)
    positive_x = float(x) + h
    negative_x = float(x) - h
    out_x = (evaluate_Fxy(str_equ,positive_x, y) - evaluate_Fxy(str_equ,negative_x, y))/(2*h)

    positive_y = float(y) + h 
    negative_y = float(y) - h
    out_y = (evaluate_Fxy(str_equ,x,positive_y) - evaluate_Fxy(str_equ, x,negative_y))/(2*h)

    return np.array([out_x,out_y])


#Deferencias finitas progresivas para derivadas 
def evaluate_derivate_fx(str_equ, x, h):
    h = float(h)
    new_x = float(x)+float(h)
    piv_out = evaluate_Fx(str_equ, new_x)
    out = -4*piv_out
    
    new_x = float(x)+2*float(h)
    piv_out = evaluate_Fx(str_equ, new_x)
    out = out + piv_out
    
    new_x = float(x)
    piv_out = evaluate_Fx(str_equ, new_x)
    out = out + 3 * piv_out
    
    out = -out/(2*h)
    #print(out)
    return out


#Deferencias finitas progresivas para derivadas 
def evaluate_derivate_fxy(str_equ, x, y,h):
    h = float(h)
    new_x = float(x)+float(h)
    new_y = float(y) + float(h)

    new_x_2h = float(x) + 2 * float(h)
    new_y_2h = float(y) + 2 * float(h)

    out_x = (-3 * evaluate_Fxy(str_equ,float(x), float(y)) + 4 * evaluate_Fxy(str_equ, new_x, float(y)) - evaluate_Fxy(str_equ, new_x_2h, float(y)))/(2*h)
    out_y = (-3 * evaluate_Fxy(str_equ,float(x), float(y)) + 4 * evaluate_Fxy(str_equ, float(x), new_y) - evaluate_Fxy(str_equ, float(x), new_y_2h))/(2*h)

    return np.array([out_x, out_y])



# Diferencias finitas centradas v2
def finite_centered_differences_v2(str_equ, x, h):
    h = float(h)
    positive_x = float(x) + h
    negative_x = float(x) - h
  
    positive_x_2h = float(x) + 2*h
    negative_x_2h = float(x) - 2*h
   

    out = (evaluate_Fx(str_equ, negative_x_2h) - 8 * evaluate_Fx(str_equ, negative_x) + 8 * evaluate_Fx(str_equ,positive_x) - evaluate_Fx(str_equ,positive_x_2h))/(12 * h)

    return out


# Diferencias finitas centradas v2
def finite_centered_differences_v2_xy(str_equ, x, y, h):
    h = float(h)
    x = float(x)
    y = float(y)
    positive_x = float(x) + h
    negative_x = float(x) - h
    positive_y = float(y) + h
    negative_y = float(y) - h
    positive_x_2h = float(x) + 2*h
    negative_x_2h = float(x) - 2*h
    positive_y_2h = float(y) + 2*h
    negative_y_2h = float(y) - 2*h

    out_x = (evaluate_Fxy(str_equ,negative_x_2h,y) - 8 * evaluate_Fxy(str_equ,negative_x,y) + 8 * evaluate_Fxy(str_equ,positive_x,y) - evaluate_Fxy(str_equ,positive_x_2h,y))/(12 * h)
    out_y = (evaluate_Fxy(str_equ,x,negative_y_2h) - 8 * evaluate_Fxy(str_equ,x,negative_y) + 8 * evaluate_Fxy(str_equ,x,positive_y) - evaluate_Fxy(str_equ,x,positive_y_2h))/(12 * h)

    return np.array([out_x,out_y])




def derivate_approx_comparisons(str_equ, x, h_range, test_h_cases):
    x = float(x)
    h_range = float(h_range)
    test_h_cases = int(test_h_cases)
    algorithm_1_list = []
    algorithm_2_list = []
    algorithm_3_list = []

    test_h = np.linspace(h_range, 0.00001,test_h_cases)

    for h_i in test_h:
        algorithm_1_list.append(np.round(finite_centered_differences(str_equ, x, h_i),6))
        algorithm_2_list.append(np.round(evaluate_derivate_fx(str_equ, x, h_i),6))
        algorithm_3_list.append(np.round(finite_centered_differences_v2(str_equ, x, h_i),6))

    parsed_real_derivative = real_derivative_Fx(str_equ)

    real_derivative = np.round(eval(parsed_real_derivative),6)

    real_derivative_list = [real_derivative for i in range(0,len(algorithm_1_list))]

    Table_1 = pd.DataFrame({'h': test_h, 'diff_finit_centradas': algorithm_1_list,'derivada real':real_derivative_list})
    Table_2 = pd.DataFrame({'h': test_h, 'diff_finit_progresivas': algorithm_2_list, 'derivada real':real_derivative_list})
    Table_3 = pd.DataFrame({'h': test_h, 'diff_finit_centradas_v2': algorithm_3_list,'derivada real':real_derivative_list})

    Table_1['error'] = np.abs(Table_1['derivada real'] - Table_1['diff_finit_centradas']) 

    Table_2['error'] = np.abs(Table_2['derivada real'] - Table_2['diff_finit_progresivas'])

    Table_3['error'] = np.abs(Table_3['derivada real'] - Table_3['diff_finit_centradas_v2']) 
    
    Tabla_4 = pd.DataFrame({'h': test_h,'Real derivative':real_derivative_list, 'Centered finite differences 2p': algorithm_1_list,'Error 1':Table_1['error'] ,
                            'Progressive finite differences 3p': algorithm_2_list,'Error 2':Table_2['error'],
                            'Centered finite differences 4p': algorithm_3_list,'Error 3':Table_3['error']})

    #return [Table_1,Table_2, Table_3]
    return Tabla_4

def derivate_approx_comparisons_xy(str_equ, x, y, h_range, test_h_cases):
    x = float(x)
    y = float(y)
    h_range = float(h_range)
    test_h_cases = int(test_h_cases)
    algorithm_1_list = []
    algorithm_2_list = []
    algorithm_3_list = []

    test_h = np.linspace(h_range, 0.00001,test_h_cases)

    for h_i in test_h:
        algorithm_1_list.append(np.round(finite_centered_differences_xy(str_equ, x, y,h_i),6))
        algorithm_2_list.append(np.round(evaluate_derivate_fxy(str_equ, x, y, h_i),6))
        algorithm_3_list.append(np.round(finite_centered_differences_v2_xy(str_equ, x, y,h_i),6))

    parsed_real_derivative_x, parsed_real_derivative_y = real_derivative_Fxy(str_equ)

    algorithm_1_list_str = []
    for item in algorithm_1_list:
        item_str = '['
        cnt = 0
        for item_i in item:
            if cnt ==0:
                item_str += str(item_i) 
            else:
                item_str += ','+str(item_i) 
            cnt += 1
        item_str += ']' 
        algorithm_1_list_str.append(item_str)
        
    algorithm_2_list_str = []
    for item in algorithm_2_list:
        item_str = '['
        cnt = 0
        for item_i in item:
            if cnt ==0:
                item_str += str(item_i) 
            else:
                item_str += ','+str(item_i) 
            cnt += 1
        item_str += ']' 
        algorithm_2_list_str.append(item_str)
        
    algorithm_3_list_str = []
    for item in algorithm_3_list:
        item_str = '['
        cnt = 0
        for item_i in item:
            if cnt ==0:
                item_str += str(item_i) 
            else:
                item_str += ','+str(item_i) 
            cnt += 1
        item_str += ']' 
        algorithm_3_list_str.append(item_str)

        
        

    real_derivative_x = np.round(eval(parsed_real_derivative_x),6)
    real_derivative_y = np.round(eval(parsed_real_derivative_y),6)
    real_derivative_list = [np.array([real_derivative_x, real_derivative_y]) for i in range(len(algorithm_1_list))]

    real_derivative_list_str = []
    for item in real_derivative_list:
        item_str = '['
        cnt = 0
        for item_i in item:
            if cnt ==0:
                item_str += str(item_i) 
            else:
                item_str += ','+str(item_i) 
            cnt += 1
        item_str += ']' 
        real_derivative_list_str.append(item_str)
        
    error_1 = np.array(algorithm_1_list) - np.array(real_derivative_list)
    error_1 = np.linalg.norm(error_1, axis = 1)
    error_2 = np.array(algorithm_2_list) - np.array(real_derivative_list)
    error_2 = np.linalg.norm(error_2, axis = 1)
    error_3 = np.array(algorithm_3_list) - np.array(real_derivative_list)
    error_3 = np.linalg.norm(error_3, axis = 1)

    Table_1 = pd.DataFrame({'h': test_h, 'Centered finite differences 2p': algorithm_1_list_str,'Real derivative':real_derivative_list_str, 'error': error_1})
    Table_2 = pd.DataFrame({'h': test_h, 'Progressive finite differences 3p': algorithm_2_list_str, 'Real derivative':real_derivative_list_str, 'error': error_2})
    Table_3 = pd.DataFrame({'h': test_h, 'Centered finite differences 4p': algorithm_3_list_str,'Real derivative':real_derivative_list_str, 'error':error_3})

    Tabla_4 = pd.DataFrame({'h': test_h,'Real derivative':real_derivative_list_str, 'Centered finite differences 2p': algorithm_1_list_str,
                            'Error 1': error_1,'Progressive finite differences 3p': algorithm_2_list_str, 'Error 2': error_2,
                            'Centered finite differences 4p': algorithm_3_list_str,'Error 3':error_3})
                
                            
    #print(Table_1)
    #print(Table_2)
    #print(Table_3)

    #return [Table_1,Table_2, Table_3]
    return Tabla_4

#derivate_approx_comparisons_xy("2x^3y^3 - 3y^2 + 3x^2 + xy + 1", 0.1, 1, 5, 25)



#Resolverdor de Newton
def newtonSolverX(x0, f_x, eps,K_max):
    x0 = float(x0)
    eps = float(eps)
    K_max = int(K_max)
    xn = x0
    error = 1
    arrayIters = []
    arrayF_x = []
    arrayf_x = []
    arrayXn = []
    arrayErr = []
    
    i = 0
    h = 0.000001
    while error > eps and K_max > i:
        print("...")
        x_n1 = xn - (evaluate_Fx(f_x, xn)/evaluate_derivate_fx(f_x, xn, h))
        error = abs(evaluate_Fx(f_x, x_n1))
        i = i + 1
        xn = x_n1
        arrayIters.append(i)
        arrayXn.append(xn)
        arrayErr.append(error)
        solution = [i, xn, error]

    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xn':arrayXn, 'Error': arrayErr})
    return TableOut

def add(a, b):
    a = int(a)
    b = int(b)
    resultado = a + b
    return "El resultado es: " + str(resultado)
  
def bisectSolverX(a,b, f_x, eps,K_max):
    #print(a,b, f_x, eps,K_max)
    a = float(a)
    b = float(b)
    K_max = int(K_max)
    eps = float(eps)
    xn = (a+b)/2
    error = 1
    arrayIters = []
    arrayF_x = []
    arrayf_x = []
    arraya = []
    arrayb = []
    arrayXn = []
    arrayErr = []
    
    i = 0
    while(error > eps and i<K_max):
        print("...")
        f_a = evaluate_Fx(f_x, a)
        f_b = evaluate_Fx(f_x, b)
        f_xn = evaluate_Fx(f_x, xn)
        if f_a * f_xn < 0:
            b = xn
        else:
            a = xn
        x_n1 = (a+b)/2
        error = abs(evaluate_Fx(f_x, xn))
        i = i + 1
        arrayIters.append(i)
        arraya.append(a)
        arrayb.append(b)
        arrayXn.append(xn)
        arrayErr.append(error)
        xn = x_n1
        solution = [i, xn, error]
  
    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xn':arrayXn,'a':arraya,'b':arrayb, 'Error': arrayErr})
    return TableOut
  
  
def gradf_QP(Q,c,x):
    return np.matmul(Q, x)+ c

def get_ESS(Q,c,x):
    num = np.dot(gradf_QP(Q,c,x),gradf_QP(Q,c,x))
    den = np.dot(gradf_QP(Q,c,x),np.matmul(Q,gradf_QP(Q,c,x)))
    return num/den

def GD_QP_Solver_ESS(x0,Q,c, eps,K_max):
    #print(a,b, f_x, eps,K_max)
    vect_piv = Q.split(',')
    len_piv = len(vect_piv)
    if len_piv **0.5 - int(len_piv **0.5) == 0.0:
        #print('Se puede')
        mat_Q = np.array(vect_piv,dtype=np.float32).reshape((int(len_piv **0.5),int(len_piv **0.5)))
    vect_piv = c.split(',')
    vec_c = np.array(vect_piv,dtype=np.float32)   
    vect_piv = x0.split(',')
    vec_x0 = np.array(vect_piv,dtype=np.float32) 
    K_max = int(K_max)
    eps = float(eps)
    xk = vec_x0
    error = 1
    arraySteps = []
    arrayIters = []
    arrayDir = []
    arrayXk = []
    arrayErr = []
    
    k = 0
    while(error > eps and k<K_max):
        print("...")
        step_k = get_ESS(mat_Q,vec_c,xk) 
        dir_k = gradf_QP(mat_Q,vec_c,xk)
        x_k1 = xk- step_k * dir_k
        dir_k1 = gradf_QP(mat_Q,vec_c,x_k1)
        error = np.linalg.norm(dir_k1)
        k = k + 1
        x_k1_str = '['
        cnt = 0
        for i in x_k1:
            if cnt ==0:
                x_k1_str += str(i) 
            else:
                x_k1_str += ','+str(i) 
            cnt += 1
        x_k1_str += ']'
        dir_k1_str = '['
        cnt = 0
        for i in dir_k1:
            if cnt ==0:
                dir_k1_str += str(i) 
            else:
                dir_k1_str += ','+str(i) 
            cnt += 1
        dir_k1_str += ']'
        arrayIters.append(k)
        arrayXk.append(x_k1_str)
        #arrayXk.append(x_k1)
        arrayErr.append(error)
        arrayDir.append(dir_k1_str)
        #arrayDir.append(dir_k1)
        arraySteps.append(step_k)
        xk = x_k1
        solution = [i, xk, error]
  
    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xk':arrayXk,'Dir':arrayDir, 'Error': arrayErr,'StepSize':arraySteps})
    return TableOut
    
    
def GD_QP_Solver_CSS(x0,Q,c, eps,K_max,css):
    #print(a,b, f_x, eps,K_max)
    vect_piv = Q.split(',')
    len_piv = len(vect_piv)
    if len_piv **0.5 - int(len_piv **0.5) == 0.0:
        #print('Se puede')
        mat_Q = np.array(vect_piv,dtype=np.float32).reshape((int(len_piv **0.5),int(len_piv **0.5)))
    vect_piv = c.split(',')
    vec_c = np.array(vect_piv,dtype=np.float32)   
    vect_piv = x0.split(',')
    vec_x0 = np.array(vect_piv,dtype=np.float32) 
    K_max = int(K_max)
    eps = float(eps)
    css = float(css)
    xk = vec_x0
    error = 1
    arraySteps = []
    arrayIters = []
    arrayDir = []
    arrayXk = []
    arrayErr = []
    
    k = 0
    while(error > eps and k<K_max):
        print("...")
        step_k = css 
        dir_k = gradf_QP(mat_Q,vec_c,xk)
        x_k1 = xk- step_k * dir_k
        dir_k1 = gradf_QP(mat_Q,vec_c,x_k1)
        error = np.linalg.norm(dir_k1)
        k = k + 1
        x_k1_str = '['
        cnt = 0
        for i in x_k1:
            if cnt ==0:
                x_k1_str += str(i) 
            else:
                x_k1_str += ','+str(i) 
            cnt += 1
        x_k1_str += ']'
        dir_k1_str = '['
        cnt = 0
        for i in dir_k1:
            if cnt ==0:
                dir_k1_str += str(i) 
            else:
                dir_k1_str += ','+str(i) 
            cnt += 1
        dir_k1_str += ']'
        arrayIters.append(k)
        arrayXk.append(x_k1_str)
        #arrayXk.append(x_k1)
        arrayErr.append(error)
        arrayDir.append(dir_k1_str)
        #arrayDir.append(dir_k1)
        arraySteps.append(step_k)
        xk = x_k1
        solution = [i, xk, error]
  
    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xk':arrayXk,'Dir':arrayDir, 'Error': arrayErr,'StepSize':arraySteps})
    return TableOut    
    
def GD_QP_Solver_VSS(x0,Q,c, eps,K_max):
    #print(a,b, f_x, eps,K_max)
    vect_piv = Q.split(',')
    len_piv = len(vect_piv)
    if len_piv **0.5 - int(len_piv **0.5) == 0.0:
        #print('Se puede')
        mat_Q = np.array(vect_piv,dtype=np.float32).reshape((int(len_piv **0.5),int(len_piv **0.5)))
    vect_piv = c.split(',')
    vec_c = np.array(vect_piv,dtype=np.float32)   
    vect_piv = x0.split(',')
    vec_x0 = np.array(vect_piv,dtype=np.float32) 
    K_max = int(K_max)
    eps = float(eps)
    xk = vec_x0
    error = 1
    arraySteps = []
    arrayIters = []
    arrayDir = []
    arrayXk = []
    arrayErr = []
    
    k = 0
    while(error > eps and k<K_max):
        print("...")
        step_k = 1/(k+1) 
        dir_k = gradf_QP(mat_Q,vec_c,xk)
        x_k1 = xk- step_k * dir_k
        dir_k1 = gradf_QP(mat_Q,vec_c,x_k1)
        error = np.linalg.norm(dir_k1)
        k = k + 1
        x_k1_str = '['
        cnt = 0
        for i in x_k1:
            if cnt ==0:
                x_k1_str += str(i) 
            else:
                x_k1_str += ','+str(i) 
            cnt += 1
        x_k1_str += ']'
        dir_k1_str = '['
        cnt = 0
        for i in dir_k1:
            if cnt ==0:
                dir_k1_str += str(i) 
            else:
                dir_k1_str += ','+str(i) 
            cnt += 1
        dir_k1_str += ']'
        arrayIters.append(k)
        arrayXk.append(x_k1_str)
        #arrayXk.append(x_k1)
        arrayErr.append(error)
        arrayDir.append(dir_k1_str)
        #arrayDir.append(dir_k1)
        arraySteps.append(step_k)
        xk = x_k1
        solution = [i, xk, error]
  
    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xk':arrayXk,'Dir':arrayDir, 'Error': arrayErr,'StepSize':arraySteps})
    return TableOut  

def gradf_RosFun(xk):
    x1 = xk[0]
    x2 = xk[1]
    c1 = 400 * x1**3 - 400 * x1 * x2 + 2 * x1 -2 
    c2 = 200 * x2 - 200 * x1**2
    return(np.array([c1,c2]))

def GD_RosFun(x0 = '0,0'):  
    vect_piv = x0.split(',')
    vec_x0 = np.array(vect_piv,dtype=np.float32) 
    print(vec_x0)
    K_max = 1000
    eps = 0.00000001
    css = 0.05
    xk = vec_x0
    error = 1
    arrayIters = []
    arrayDir = []
    arrayXk = []
    arrayErr = []
    
    k = 0
    while(error > eps and k<K_max):
        print("...")
        step_k = css 
        dir_k = gradf_RosFun(xk)
        x_k1 = xk- step_k * dir_k
        dir_k1 = gradf_RosFun(x_k1)
        error = np.linalg.norm(dir_k1)
        k = k + 1
        x_k1_str = '['
        cnt = 0
        for i in x_k1:
            if cnt ==0:
                x_k1_str += str(i) 
            else:
                x_k1_str += ','+str(i) 
            cnt += 1
        x_k1_str += ']'
        dir_k1_str = '['
        cnt = 0
        for i in dir_k1:
            if cnt ==0:
                dir_k1_str += str(i) 
            else:
                dir_k1_str += ','+str(i) 
            cnt += 1
        dir_k1_str += ']'
        arrayIters.append(k)
        arrayXk.append(x_k1_str)
        #arrayXk.append(x_k1)
        arrayErr.append(error)
        arrayDir.append(dir_k1_str)
        #arrayDir.append(dir_k1)
        xk = x_k1
        solution = [i, xk, error]
  
    print("Finalizo...")
    TableOut = pd.DataFrame({'Iter':arrayIters, 'Xk':arrayXk,'Dir':arrayDir, 'Error': arrayErr})
    return TableOut  
    
    
    
def generateDataset():
    d  = 100 # cantidad de columnas para el dataset
    n = 1000
    np.random.seed(42) 
    A = np.random.normal(0,1,size = (n,d))
    x_true = np.random.normal(0,1, size = (d,1))
    b = A.dot(x_true) + np.random.normal(0,0.5, size = (n,1))

    return A,b

def closed_solution_linear_regression():

    A, b = generateDataset()
    total_examples = A.shape[0]
    
    optimal_x = np.linalg.inv(np.matmul(A.T, A)) @ np.matmul(A.T, b)

    predictions = A.dot(optimal_x) 

    error = np.sum((predictions - b)**2)

    return optimal_x, error 


#print(closed_solution_linear_regression())

def gradient_descent(epochs):

    A, b = generateDataset()
    #print(A.shape)

    total_examples = A.shape[0]

    x = np.zeros(shape = (A.shape[1],1))

    #print(x)

    predictions = A.dot(x)
    error = np.sum((predictions - b)**2)

    #print(error)

    dx_df = np.dot(A.T, (predictions - b)) / total_examples

    #print(dx_df)

    all_errors = []

    step_sizes_to_test = [0.00005, 0.0005, 0.0007]

    for step_size in step_sizes_to_test:
        #print(step_size)
        x = np.zeros(shape = (A.shape[1],1))
        error_list = []

        for step in range(0,epochs):
            x = x - (step_size * dx_df)

            predictions = A.dot(x)
            error = np.sum((predictions - b)**2)

            dx_df = np.dot(A.T, 2*(predictions - b))

            error_list.append(error)

        all_errors.append(np.array(error_list))

    return all_errors
        

#print(gradient_descent(20))
