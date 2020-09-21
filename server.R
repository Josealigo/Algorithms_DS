
library(shiny)
library(reticulate)
source_python('algoritmos.py')

shinyServer(function(input, output) {
    
    #Evento y evaluación de metodo de newton para ceros
    newtonCalculate<-eventReactive(input$nwtSolver, {
        print(c('Estamos en Newton'))
        inputEcStr<-input$equation_New[1]
        #print(inputEcStr)
        initVal<-input$initVal_New[1]
        error<-input$Error_New[1]
        MaxIter<-input$MaxIter_New[1]
        #print(MaxIter)
        #outs<-add(initVal, error)
        outs<-newtonSolverX(initVal, inputEcStr, error,MaxIter)
        print(outs)
        outs
    })
    
    #Evento y evaluación del metodo de la biseccion
    bisectionCalculate<-eventReactive(input$bisSolver, {
        #print(c('Estamos en Bisection'))
        #print(input)
        inputEcStr<-input$equation_Bis[1]
        #print(c('Equation:',inputEcStr))
        initVal_a<-input$initVal_a_Bis[1]
        #print(c('initVal_a:',initVal_a))
        initVal_b<-input$initVal_b_Bis[1]
        #print(c('initVal_b:',initVal_b))
        error<-input$Error_Bis[1]
        print(c('error:',error))
        MaxIter<-input$MaxIter_Bis[1]
        print(c('MaxIter:',MaxIter))
        #outs<-add(initVal, error)
        outs<-bisectSolverX(initVal_a,initVal_b, inputEcStr, error,MaxIter)
        outs
    })
    
    #Evento y evaluación del metodo de gradiente para QP ESS 
    GD_QP_ESS_Calculate<-eventReactive(input$GD_QP_ESS, {
        #print(c('Estamos en GD QP ESS'))
        Init_X <- input$Init_GDQP
        #print(Init_X)
        Matrix_GDQP<-input$Matrix_GDQP
        #print(c('Matrix_GDQP:',Matrix_GDQP))
        Vector_GDQP<-input$Vector_GDQP
        #print(c('Vector_GDQP:',Vector_GDQP))
        error<-input$Error_GDQP[1]
        #print(c('error:',error))
        MaxIter<-input$MaxIter_GDQP[1]
        #print(c('MaxIter:',MaxIter))
        outs = 0
        #print(c('Entramos a GD_QP_Solver'))
        #outs<-GD_QP_Solver_ESS(1,2,3, 4,5)
        outs<-GD_QP_Solver_ESS(Init_X,Matrix_GDQP,Vector_GDQP, error,MaxIter)
        #print(c('Salimos de GD_QP_Solver'))
        #print(outs)
        outs
    })
    
    #Evento y evaluación del metodo de gradiente para QP CSS 
    GD_QP_CSS_Calculate<-eventReactive(input$GD_QP_CSS, {
        #print(c('Estamos en GD QP ESS'))
        Init_X <- input$Init_GDQP
        #print(Init_X)
        Matrix_GDQP<-input$Matrix_GDQP
        #print(c('Matrix_GDQP:',Matrix_GDQP))
        Vector_GDQP<-input$Vector_GDQP
        #print(c('Vector_GDQP:',Vector_GDQP))
        error<-input$Error_GDQP[1]
        #print(c('error:',error))
        MaxIter<-input$MaxIter_GDQP[1]
        #print(c('MaxIter:',MaxIter))
        CSS<-input$CSS_GDQP[1]
        #print(c('CSS:',CSS))
        outs = 0
        #print(c('Entramos a GD_QP_Solver'))
        outs<-GD_QP_Solver_CSS(Init_X,Matrix_GDQP,Vector_GDQP, error,MaxIter,CSS)
        #print(c('Salimos de GD_QP_Solver'))
        #print(outs)
        outs
    })
    
    #Evento y evaluación del metodo de gradiente para QP VSS 
    GD_QP_VSS_Calculate<-eventReactive(input$GD_QP_VSS, {
        #print(c('Estamos en GD QP ESS'))
        Init_X <- input$Init_GDQP
        #print(Init_X)
        Matrix_GDQP<-input$Matrix_GDQP
        #print(c('Matrix_GDQP:',Matrix_GDQP))
        Vector_GDQP<-input$Vector_GDQP
        #print(c('Vector_GDQP:',Vector_GDQP))
        error<-input$Error_GDQP[1]
        #print(c('error:',error))
        MaxIter<-input$MaxIter_GDQP[1]
        #print(c('MaxIter:',MaxIter))
        #print(c('Entramos a GD_QP_Solver'))
        outs<-GD_QP_Solver_VSS(Init_X,Matrix_GDQP,Vector_GDQP, error,MaxIter)
        #print(c('Salimos de GD_QP_Solver'))
        #print(outs)
        outs
    })
    
    #Evento y evaluación del metodo de gradiente para RosFun
    GD_RosFun_Calculate<-eventReactive(input$GD_RosFun, {
        #print(c('Estamos en GD para RosFun'))
        Init_X <- input$Init_GD_RosFun
        #print(Init_X)
        #print(c('Entramos a GD_RosFun'))
        outs<-GD_RosFun(Init_X)
        #print(c('Salimos de GD_RosFun'))
        #print(outs)
        outs
    })
    
    #Evento y evaluación de diferencias finitas centradas 2
    diferFinitCalculatec2<-eventReactive(input$diferFinEval2, {
        inputEcStr<-input$equation_CDi2[1]
        valX<-input$valX_CDi2[1]
        h<-input$valH_CDi2[1]
        outs<-finite_centered_differences(inputEcStr, valX, h)
        as.character(outs)
    })    
    
    #Evento y evaluación de diferencias finitas centradas 4
    diferFinitCalculatec4<-eventReactive(input$diferFinEval4, {
        inputEcStr<-input$equation_CDi4[1]
        valX<-input$valX_CDi4[1]
        h<-input$valH_CDi4[1]
        outs<-finite_centered_differences_v2(inputEcStr, valX, h)
        as.character(outs)
    })    
    
    #Evento y evaluación de diferencias finitas progresivas
    diferFinitCalculate<-eventReactive(input$diferFinEval, {
        inputEcStr<-input$equation_FDi[1]
        valX<-input$valX_FDi[1]
        h<-input$valH_FDi[1]
        outs<-evaluate_derivate_fx(inputEcStr, valX, h)
        as.character(outs)
    })
    
    #Evento y evaluación de regresion lineal con solucion cerrada
    GD_1_Sol_Calculate<-eventReactive(input$GD_1_Sol, {
        outs<-closed_solution_linear_regression()
        outs
    })
    
    #Evento y evaluación de regresion lineal con GD
    GD_2_Sol_Calculate<-eventReactive(input$GD_2_Sol, {
        SS<-input$GD_2_SS[1]
        outs<-gradient_descent_v2(20,SS)
        outs
    })
    
    #Evento y evaluación de Newton en 2D
    newtonCalculate2D<-eventReactive(input$nwtSolver2D, {
        #print(c('Estamos en Newton'))
        initValx<-input$initValx_New2D[1]
        initValy<-input$initValy_New2D[1]
        outs<-NS_RosFun(initValx,initValy)
        #print(outs)
        outs
    })
    
    
    
    
    #REnder metodo de Newton
    output$salidaNewt<-renderTable({
        newtonCalculate()
    }, digits = 5)
    
    #REnder metodo de Biseccion
    output$salidaBisec<-renderTable({
        bisectionCalculate()
    }, digits = 5)
    
    #Render Diferncias Finitas progresivas
    output$difFinitOut<-renderText({
        diferFinitCalculate()
    })
    
    #Render Diferncias Finitas centradas 2
    output$difFinitOutc2<-renderText({
        diferFinitCalculatec2()
    })
    
    #Render Diferncias Finitas centradas 4
    output$difFinitOutc4<-renderText({
        diferFinitCalculatec4()
    })
    
    #Render piv_text_GD2_1
    output$piv_text_GD2_1<-renderTable({
        Tabla = gradient_descent(20)
        Tabla[10,3]
        #'a'
    })
    
    #Render GD QP ESS
    output$GD_QP_ESS_Output<- renderTable({
        GD_QP_ESS_Calculate()
    }, sdigits = 5, caption="Exact Step Size",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #Render GD QP CSS
    output$GD_QP_CSS_Output<- renderTable({
        GD_QP_CSS_Calculate()
    }, sdigits = 5, caption="Constant Step Size",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #Render GD QP VSS
    output$GD_QP_VSS_Output<- renderTable({
        GD_QP_VSS_Calculate()
    }, sdigits = 5, caption="Variable Step Size",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #Render GD_RosFun_Output
    output$GD_RosFun_Output<- renderTable({
        GD_RosFun_Calculate()
    }, sdigits = 5)
    
    
    #REnder metodo de Newton
    output$salidaNewt2D<-renderTable({
        newtonCalculate2D()
    }, digits = 5)
    
    #Render GD_1_Sol_Output
    output$GD_1_Sol_Output<- renderTable({
        Tabla = GD_1_Sol_Calculate()
        Tabla[1]
    }, sdigits = 5)
    
    #Render GD_1_Error_Output
    output$GD_1_Error_Output<- renderTable({
        Tabla = GD_1_Sol_Calculate()
        Tabla[2]
    }, sdigits = 5)
    
    # Render plot para mostrar la diferencia de Newton con Biseccion 
    output$plot1 <- renderPlot({
        plot(seq(-1,1,0.1), exp(seq(-1,1,0.1))+2*seq(-1,1,0.1),
             main="F(x) = e^x+2x",
             ylab="F(x)",
             xlab = "x",
             type="l",
             col="blue",
             xlim = c(-1,1),
             ylim = c(-2,2))
    })    
    
    # Render plor para la grafica uno de regresion lineal GD_2  gradient_descent
    output$plot1_GD2 <- renderPlot({
        Tabla = gradient_descent(20)
        Tabla[,1]
        plot(Tabla[,1], ylab = "f(x)",xlab = "Iter",col = "Blue", main = " F(x) per iteration",pch = 19,ylim = c(0,200000))
        points(Tabla[,2], col= "Green",pch = 19)
        points(Tabla[,3], col = "Red",pch = 19)
        legend("topright",col = c("Blue","Green", "Red"), legend = c("0.00005","0.0005","0.0007"), pch = 19 )
    })   
    
    output$plot2_GD2 <- renderPlot({
        Data1 = GD_2_Sol_Calculate()
        ss = colnames(Data1)[1]
        plot(Data1[,1], ylab = "f(x)",xlab = "Iter",col = "Blue", main = " F(x) per iteration",pch = 19,ylim = c(0,200000))
        legend("topright",col = c("Blue"), legend = c(ss), pch = 19 )
    })   
    
    #Evento y comparasion de diferencias finitas 1D
    diferFinitComp1D<-eventReactive(input$C_FD_1D, {
        inputEcStr<-input$f_FS_1D[1]
        valX<-input$x_FD_1D[1]
        h<-input$h_FD_1D[1]
        ch<-input$ch_FD_1D[1]
        outs<-derivate_approx_comparisons(inputEcStr, valX, h, ch)
        outs
    })       
    
    #Evento y comparasion de diferencias finitas 2D
    diferFinitComp2D<-eventReactive(input$C_FD_2D, {
        inputEcStr<-input$f_FS_2D[1]
        valX<-input$x_FD_2D[1]
        valy<-input$y_FD_2D[1]
        h<-input$h_FD_2D[1]
        ch<-input$ch_FD_2D[1]
        outs<-derivate_approx_comparisons_xy(inputEcStr, valX, valy, h, ch)
        outs
    })    
    
    #REnder Comparacion diferencias finitas 
    output$Notes0_DC1<-renderTable({
        diferFinitComp1D()
    }, digits = 7)
    
    #REnder Comparacion diferencias finitas xy
    output$Notes0_DC2<-renderTable({
        diferFinitComp2D()
    }, digits = 7)
    
    #REnder metodo de Newton para e^x+2*x
    output$Notes1_New<-renderTable({
        newtonSolverX('0', 'e^x+2*x', 0.0001,50)
    }, digits = 5, caption="Newton's method",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #REnder metodo de Biseccion para e^x+2*x
    output$Notes1_Bis<-renderTable({
        bisectSolverX(-1,0,'e^x+2*x', 0.0001,50)
    }, digits = 5, caption="Bisection Method",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #REnder metodo de GD con ESS
    output$Notes2_ESS<-renderTable({
        GD_QP_Solver_ESS('3,5,7','2,-1,0,-1,2,-1,0,-1,2','1,0,1', '0.000001','30')
    }, digits = 5, caption="Exact Step Size",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #REnder metodo de GD con CSS
    output$Notes2_CSS1<-renderTable({
        GD_QP_Solver_CSS('3,5,7','2,-1,0,-1,2,-1,0,-1,2','1,0,1', '0.000001','30','0.1')
    }, digits = 5, caption="Constant Step Size (0.1)",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #REnder metodo de GD con CSS
    output$Notes2_CSS2<-renderTable({
        GD_QP_Solver_CSS('3,5,7','2,-1,0,-1,2,-1,0,-1,2','1,0,1', '0.000001','30','1')
    }, digits = 5, caption="Constant Step Size (1)",caption.placement = getOption("xtable.caption.placement", "top"))
    
    #REnder metodo de GD con VSS
    output$Notes2_VSS<-renderTable({
        GD_QP_Solver_VSS('3,5,7','2,-1,0,-1,2,-1,0,-1,2','1,0,1', '0.000001','30')
    }, digits = 5, caption="Variable Step Size",caption.placement = getOption("xtable.caption.placement", "top"))
    
    
})
