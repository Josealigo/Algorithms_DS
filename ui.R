library(shiny)
library(shinydashboard)

# Define UI for application that draws a histogram
dashboardPage(
    dashboardHeader(title = "Algorithms on DS"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("Makers", tabName = "Makers"),
            menuItem("Centered Finite Differences (2p)", tabName = "Derivative_1"),
            menuItem("Centered Finite Differences (4p)", tabName = "Derivative_2"),
            menuItem("Progressive Finite Differences (3p)", tabName = "Derivative_3"),
            menuItem("Newton", tabName = "Newton"),
            menuItem("Bisection", tabName = "Bisec"),
            menuItem("Gradient Descent on QP", tabName = "GD_QP"),
            menuItem("Rosenbrock's Function", tabName = "RosFun"),
            menuItem("Linear regression (Closed Solution)", tabName = "GD_1"),
            menuItem("Linear regression (GD Solve)", tabName = "GD_2"),
            menuItem("Linear regression (SGD Solve)", tabName = "GD_3"),
            menuItem("Linear regression (MBGD Solve)", tabName = "GD_4"),
            menuItem("Linear regression Solutions", tabName = "GD_5"),
            menuItem("Newton on Rosenbrock's F", tabName = "Newton2D"),
            menuItem("Notes 0_1", tabName = "Notes01"),
            menuItem("Notes 0_2", tabName = "Notes02"),
            menuItem("Notes 1", tabName = "Notes1"),
            menuItem("Notes 2", tabName = "Notes2")
            # 
            
        )
    ),
    dashboardBody(
        tabItems(
            tabItem("Makers",
                    h1("Members"),
                    box(column("Juan Pablo Carranza", status = "primary", 
                               title = "1",  width = 6 , height = 120 ),
                        column("Jose Alberto Ligorria", status = "primary", 
                               title = "2",  width = 6 , height = 120 ))
                    ),
            
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
            
            tabItem("Derivative_1",
                    h1("Centered Finite Differences on two points"),
                    box(textInput("equation_CDi2", "Equation"),
                        textInput("valX_CDi2", "x"),
                        textInput("valH_CDi2", "h")),
                    actionButton("diferFinEval2", "Evaluate Derivative"),
                    textOutput("difFinitOutc2")),
            
            tabItem("Derivative_2",
                    h1("Centered Finite Differences on four points"),
                    box(textInput("equation_CDi4", "Equation"),
                        textInput("valX_CDi4", "x"),
                        textInput("valH_CDi4", "h")),
                    actionButton("diferFinEval4", "Evaluate Derivative"),
                    textOutput("difFinitOutc4")),
            
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
                    box(textInput("Init_GDQP", "X0",value = "3,5,7"),
                        textInput("Matrix_GDQP", "Q Matrix", "2,-1,0,-1,2,-1,0,-1,2"),
                        textInput("Vector_GDQP", "c vector","1,0,1"),
                        textInput("Error_GDQP", "Error","0.000001"),
                        textInput("MaxIter_GDQP", "Max iterations","30"),
                        checkboxGroupInput("CSS_GDQP","Constant Step Size",choices = c(0.001, 0.01, 0.1, 0.5, 1))),
                    actionButton("GD_QP_ESS", "Exact Step Size"),
                    actionButton("GD_QP_CSS", "Constant Step Size"),
                    actionButton("GD_QP_VSS", "Variable Step Size"),
                    tableOutput("GD_QP_ESS_Output"),
                    tableOutput("GD_QP_CSS_Output"),
                    tableOutput("GD_QP_VSS_Output")),
            
            tabItem("RosFun",
                    h1("Gradient Descent on Rosenbrock's Function"),
                    h3('The method is not convergent with the combination of step size and initial value, lots of initial value where tested, perhaps because GD is not capable of it because of the scale. You are welcome to try on'),
                    box(textInput("Init_GD_RosFun", "X0","0,0")),
                    actionButton("GD_RosFun", "Gradient Descent"),
                    tableOutput("GD_RosFun_Output")),
            
            tabItem("GD_1",
                    h1("Closed Solution on Linear Regression"),
                    h3('We will consider the problem of adjusting a regression line to a data set. The regression model is given by the equation Ax = b, where A represents the set of inputs of dimension n × d and b represents the set of observations of dimension n × 1. The objective is to determine the vector of coefficients x of dimension d × 1 and that minimizes the sum of squared errors, The dataset is generated with the next function:'),
                    tags$img(src='RL_Dataset.png'),
                    h3('The first solution is given by a closed function, which gives the data aside.'),
                    h3('It is not the best to opt for this solution because doing the multiplication of matrices in a large data set is expensive and nothing guarantees that the matrix obtained is invertible.'),
                    actionButton("GD_1_Sol", "Get Solution"),
                    tableOutput("GD_1_Error_Output"),
                    tableOutput("GD_1_Sol_Output")),
            
            tabItem("GD_2",
                    h1("Gradient Descent Solution on Linear Regression"),
                    h3('We will consider the problem of adjusting a regression line to a data set. The regression model is given by the equation Ax = b, where A represents the set of inputs of dimension n × d and b represents the set of observations of dimension n × 1. The objective is to determine the vector of coefficients x of dimension d × 1 and that minimizes the sum of squared errors, The dataset is generated with the next function:'),
                    tags$img(src='RL_Dataset.png'),
                    h3('The second solution is given by Gradient Descent, which gives the data aside.'),
                    box(column(plotOutput("plot1_GD2", click = "plot_click",width = "150%",height = "400px"),
                               status = "primary", 
                               title = "O",  width = 6 , height = 120)),
                        column(textInput("GD_2_SS", "Step size","0.001"),
                               actionButton("GD_2_Sol", "Get Solution"),
                               plotOutput("plot2_GD2", click = "plot_click",width = "90%",height = "400px"),
                               box(h4("After training with various values, it was found that before 0.0004 the convergence is slower and after 0.0005 it is also slower until the divergence, so the best value of those tried was 0.00045."),width =6), status = "primary", 
                               title = "O",  width = 6 , height = 120)
                    ),
            
            tabItem("GD_3",
                    h1("Stochastic Gradient Descent Solution on Linear Regression"),
                    h3('We will consider the problem of adjusting a regression line to a data set. The regression model is given by the equation Ax = b, where A represents the set of inputs of dimension n × d and b represents the set of observations of dimension n × 1. The objective is to determine the vector of coefficients x of dimension d × 1 and that minimizes the sum of squared errors, The dataset is generated with the next function:'),
                    tags$img(src='RL_Dataset.png'),
                    h3('The third solution is given by Stochastic Gradient Descent, which gives the data aside.'),
                    actionButton("GD_3_Sol", "Get Solution"),
                    plotOutput("plot1_GD3", click = "plot_click",width = "90%",height = "400px"),
                    box(h4('For stochastic gradient descent none of the given step sizes converged to any of the answers, Additional ones were tried but none of them worked too. It may be related to the specific random dataset generated by the random seed set.'))
                    ),
            
            tabItem("GD_4",
                    h1("Mini Batch Gradient Descent Solution on Linear Regression"),
                    h3('We will consider the problem of adjusting a regression line to a data set. The regression model is given by the equation Ax = b, where A represents the set of inputs of dimension n × d and b represents the set of observations of dimension n × 1. The objective is to determine the vector of coefficients x of dimension d × 1 and that minimizes the sum of squared errors, The dataset is generated with the next function:'),
                    tags$img(src='RL_Dataset.png'),
                    h3('The fourth solution is given by Mini Batch Gradient Descent, which gives the data aside.'),
                    actionButton("GD_4_Sol", "Get Solution"),
                    box(column(plotOutput("plot1_GD4", click = "plot_click",width = "90%",height = "400px"),  width = 4 , height = 120 ),
                        column(plotOutput("plot2_GD4", click = "plot_click",width = "90%",height = "400px"),  width = 4 , height = 120 ),
                        column(plotOutput("plot3_GD4", click = "plot_click",width = "90%",height = "400px"),  width = 4 , height = 120 ),width =NULL),
                    box(h4('For Mini Batch gradient descent, the best of all the combinations of batch size and step size was a batch size of 50 with a step size of 0.005. This yielded the fastest convergence to an optimal value with the least variance between each iteration.'))
            ),
            
            tabItem("GD_5",
                    h1("Comparison of solutions for linear regressions"),
                    h3('According to our experiments done this is the way we order the effectiveness of the three algorithms, from best to worst: Gradient descent with a final error of 210 after 20 iterations, Mini Batch gradient descent with a final error of 514 after 5 epochs over whole dataset, and finally SGD with no convergence after 1000 iterations (experiments were done with multiple epochs over whole dataset but no convergence occured).  

In this case we conclude that gradient descent is the best option, but in cases where the computational costs become prohibitive, the best algorithm may be Mini Batch gradient Descent. 

Even though MSE is a convex function, there is no guarantee of convergence without an exact line search.'),
                    ),
            
            tabItem("Newton2D",
                    h1("Newton's method on 2D for Rosenbrock's Function"),
                    box(textInput("initValx_New2D", "X1","0"),
                        textInput("initValy_New2D", "x2","0")),
                    actionButton("nwtSolver2D", "Newton Solver"),
                    tableOutput("salidaNewt2D")),
            
            
            tabItem("Notes01",
                    h1("Finite Differences Comparisons"),
                    box(h3('We are going to see the difference on derivatives estimations for the next function.')),
                    box(textInput("f_FS_1D", "f(x)"),
                        textInput("x_FD_1D", "x"),
                        textInput("h_FD_1D", "h0",value = 1),
                        h5('Final h is set to 0.00001'),
                        textInput("ch_FD_1D", "Amount of h",value = 10)),
                    actionButton("C_FD_1D", "Make comparison"),
                    tableOutput("Notes0_DC1")),
            
            tabItem("Notes02",
                    h1("Finite Differences Comparisons on 2 dimentions"), 
                    box(h3('We are going to see the difference on derivatives estimations for the next function.')),
                    box(textInput("f_FS_2D", "f(x,y)"),
                        textInput("x_FD_2D", "x"),
                        textInput("y_FD_2D", "y"),
                        textInput("h_FD_2D", "h0",value = 1),
                        h5('Final h is set to 0.00001'),
                        textInput("ch_FD_2D", "Amount of h",value = 10)),
                    actionButton("C_FD_2D", "Make comparison"),
                    tableOutput("Notes0_DC2")),
            
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
