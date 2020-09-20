library(shiny)
library(shinydashboard)

# Define UI for application that draws a histogram
dashboardPage(
    dashboardHeader(title = "Algorithms on DS"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("Newton", tabName = "Newton"),
            menuItem("Bisection", tabName = "Bisec"),
            menuItem("Gradient Descent on QP", tabName = "GD_QP"),
            menuItem("Rosenbrock's Function", tabName = "RosFun"),
            menuItem("Progressive Finite Differences", tabName = "Derivative_3"),
            menuItem("Notes 1", tabName = "Notes1"),
            menuItem("Notes 2", tabName = "Notes2")
            
            
        )
    ),
    dashboardBody(
        tabItems(
            tabItem("Newton",
                    h1("Newton's method"),
                    box(textInput("equation_New", "Equation"),
                        textInput("initVal_New", "X0"),
                        textInput("Error_New", "Error"),
                        textInput("MaxIter_New", "Max iterations")),
                    actionButton("nwtSolver", "Newton Solver"),
                    tableOutput("salidaNewt")),
            
            tabItem("Bisec",
                    h1("Bisection Method"),
                    box(textInput("equation_Bis", "Equation"),
                        textInput("initVal_a_Bis", "a"),
                        textInput("initVal_b_Bis", "b"),
                        textInput("Error_Bis", "Error"),
                        textInput("MaxIter_Bis", "Max iterations")),
                    actionButton("bisSolver", "Bisection Solver"),
                    tableOutput("salidaBisec")),
            
            tabItem("Derivative_3",
                    h1("Progressive Finite Differences"),
                    box(textInput("equation_FDi", "Equation"),
                    textInput("valX_FDi", "x"),
                    textInput("valH_FDi", "h")),
                    actionButton("diferFinEval", "Evaluate Derivative"),
                    textOutput("difFinitOut")),
            
            tabItem("GD_QP",
                    h1("Gradient Descent on QP optimization problem"),
                    h3(HTML("Please for matrix and vector notation use coma separated values from q<sub>11</sub> to q<sub>nn</sub> through rows")),
                    box(textInput("Init_GDQP", "X0"),
                        textInput("Matrix_GDQP", "Q Matrix"),
                        textInput("Vector_GDQP", "c vector"),
                        textInput("Error_GDQP", "Error"),
                        textInput("MaxIter_GDQP", "Max iterations"),
                        checkboxGroupInput("CSS_GDQP","Constant Step Size",choices = c(0.001, 0.01, 0.1, 0.5, 1))),
                    actionButton("GD_QP_ESS", "Exact Step Size"),
                    actionButton("GD_QP_CSS", "Constant Step Size"),
                    actionButton("GD_QP_VSS", "Variable Step Size"),
                    tableOutput("GD_QP_ESS_Output"),
                    tableOutput("GD_QP_CSS_Output"),
                    tableOutput("GD_QP_VSS_Output")),
            
            tabItem("RosFun",
                    h1("Gradient Descent  on Rosenbrock's Function"),
                    h3('The method is not convergent with the combination of step size and initial value, lots of initial value where tested, perhaps because GD is not capable of it because of the scale. You are welcome to try on'),
                    box(textInput("Init_GD_RosFun", "X0","0,0")),
                    actionButton("GD_RosFun", "Gradient Descent"),
                    tableOutput("GD_RosFun_Output")),
            
            tabItem("Notes1",
                    h1("Newton versus Bisection"),
                    box(h3('We are going to find a stationary point of the function e^x+x^2.'),
                        plotOutput("plot1", click = "plot_click",width = "50%",height = "200px"),
                        h3('From the graph we can see our zeros between -1 and 0, so we will pursue it from 0 on Newtow and with -1,0 on Bisection.'),
                        h3('From the tables on the side we can see that Newtow is faster than Bisection in this case. Let\'s see somo other comparation of this methods:'),
                        h3(HTML('<b> Bisection </b>')),
                        h3(column(HTML('Advantages <br> It is always convergent. <br> It is easy to implement.'), status = "primary", 
                                  title = "O",  width = 6 , 
                                  height = 120),
                           column(HTML('Disadvantages <br> It converges very slowly. <br> It allows to find only one root, even if there are more in the interval.'), status = "primary", 
                                  title = "O",  width = 6 , 
                                  height = 120)),
                        h3(HTML('<b> Newton </b>')),
                        h3(column(HTML('Advantages <br> Its efficient <br> Works quite well with nonlinear functions'), status = "primary", 
                                  title = "O",  width = 6 , 
                                  height = 120),
                           column(HTML('Disadvantages <br> Derivative evaluation <br> Works poorly when there is an inflection point near a root'), status = "primary", 
                                  title = "O",  width = 6 , 
                                  height = 120))
                        ),
                    tableOutput("Notes1_New"),
                    tableOutput("Notes1_Bis")),
            
            tabItem("Notes2",
                    h1("Step Size on GD"),
                    box(h3('We are going to see the difference on the convergence of GD with different step sizes across the four tables on the side. The QP problem solved and the initial considerations are the ones below.'),
                        tags$img(src='QP_1.png'),
                        h3('From the tables that if the constant steep size is too large it diverges and meanwhile the variable step size reaches and leaves the exact beyond for a while it rapidly stops growing, so the step size is very important and the exact step size is a good approximation of the best GD convergence.' )),
                    tableOutput("Notes2_ESS"),
                    tableOutput("Notes2_CSS1"),
                    tableOutput("Notes2_CSS2"),
                    tableOutput("Notes2_VSS"))
        )
    )
)
