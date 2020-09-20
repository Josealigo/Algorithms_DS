
library(shiny)
library(reticulate)
# Create a virtual environment selecting your desired python version
#virtualenv_create(envname = "python_environment", python= '/usr/bin/python3')
# Explicitly install python libraries that you want to use, e.g. pandas, numpy
#virtualenv_install("python_environment", packages = c('pandas','numpy'), ignore_installed = TRUE)
# Select the virtual environment
#use_virtualenv("python_environment", required = TRUE)
                   
source_python("algoritmos.py")
#source_python("C:/Users/jose-/Documents/Maestria/Algoritmos/plataforma_base/demo1/algoritmos.py")

#tableOut, soluc = newtonSolverX(-5, "2x^5 - 3", 0.0001)

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
    
    
    
    
    
    
    #Evento y evaluación de diferencias finitas
    diferFinitCalculate<-eventReactive(input$diferFinEval, {
        inputEcStr<-input$equation_FDi[1]
        valX<-input$valX_FDi[1]
        h<-input$valH_FDi[1]
        outs<-evaluate_derivate_fx(inputEcStr, valX, h)
        as.character(outs)
    })
    
    
    #REnder metodo de Newton
    output$salidaNewt<-renderTable({
        newtonCalculate()
    }, digits = 5)
    
    #REnder metodo de Biseccion
    output$salidaBisec<-renderTable({
        bisectionCalculate()
    }, digits = 5)
    
    #Render Diferncias Finitas
    output$difFinitOut<-renderText({
        diferFinitCalculate()
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
