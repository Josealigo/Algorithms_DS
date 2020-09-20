import pandas as pd
import numpy as np
import re
import math

#Evaluación REGREX
def evaluate_Fx(str_equ, valX):
    x = float(valX)
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
    exp_chage = [a for a in re.findall(r"[0-9]\(", strOut )]
    for i in exp_chage:
        num = i.replace("(","")
        new_exp = num + '*('
        strOut = strOut.replace(i,new_exp)
    out = eval(strOut)
    return out

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
