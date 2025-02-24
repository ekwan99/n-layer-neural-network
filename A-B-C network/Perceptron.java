/*
* @author Emily Kwan
* @version 3/4/24
* 
* The Perceptron class contains code for an A-B-C network, which is a network configured to have 
* A input nodes, B hidden layer nodes, and C output nodes. This neural network trains and alters weights 
* using a gradient (steepest) descent method in order to minimize error. This network applies a sigmoid
* activation function to calculate outputs. This class also includes functionalities to configure the model 
* and print different results based on the mode it is operating in, whether that is training or running. The 
* model's test cases and expected outputs are manually set.
*/
import java.io.*;

public class Perceptron
{
/* 
* Declaration of network configuration variables  
*/ 
   private static int numIn;             // number of input activations
   private static int numHidden;         // number of hidden activations
   private static int numOut;            // number of output activations
   private static boolean train;         // identifies whether or not the model will be trained; if false, it is just ran
   private static boolean randomize;     // identifies if weights will be randomized; if false, they are manually set
   private static double errorThreshold; // the program will terminate if the error is below this threshold
   private static int numTests;          // number of test cases
   private static int i, j, k;           // iterators of the layers--output, hidden, and input layers, respectively
/*
* Declaration of arrays for processing units and weights between layers
*/
   private static double[] a;            // array of input activations
   private static double[] h;            // array of hidden activations
   private static double[][] wKJ;        // array of weights between input layer and hidden layer
   private static double[][] wJI;        // array of weights between hidden layer and output layer
   private static String weightsFile;    // name of the file that contains saved/loaded weights
/* 
* Declaration of arrays and tables with training data, running data, calculated outputs, and calculated errors
*/
   private static double[][] testingData;     // table with manually initialized values with test cases for running and training
   private static double[][] testingExpected; // expected calculated values using the given test cases
   private static double[][] F;               // array of outputs as calculated by the neural network
   private static double[] errors;            // array of errors for each output when compared to expected values
   private static int t;                      // iterator for the test cases
/* 
* Declaration of arrays used in training and calculating weights
*/
   private static double[] thetaJArray;   // array of theta values for the first layer of the network
   private static double[] thetaIArray;   // array of theta values for the second layer of the network
   private static double[][] deltaWKJ;    // array of changes in each weight between the input and hidden layers
   private static double[][] deltaWJI;    // array of changes in each weight between the hidden and output layers
   private static double[][] errorWRTwKJ; // array of errors in terms of weights between first and hidden layers
   private static double[][] errorWRTwJI; // array of errors in terms of weights between hidden and output layers
   private static double[] omegaJ;        // array of capital omega values for the first layer
   private static double[] psiJ;          // psi value for the first layer
/*
* Declaration of values and arrays used in training and calculating weights
*/
   private static double lambda;    // learning factor--controls how quickly weights are changed
   private static double minRand ;  // minimum value that randomly generated weights can be (inclusive)
   private static double maxRand;   // maximum value that randomly generated weight can be (inclusive)
   private static int maxIters;     // maximum number of iterations the program can reach
   private static boolean done;     // identifies whether or not the program is done training or not
   private static double thetaJ;    // theta value for the first layer
   private static double thetaI;    // theta value for the second layer
   private static double[] psiI;    // psi value for the second layer
   private static double[] omegaI;  // omega value for the second layer
/*
* Declaration of variables used to report results
*/
   private static int iterationsReached;   // amount of iterations the program reached at termination
   private static String reasonToEndTrain; // reason for training end: either max. iter. reached, or avg. error is below threshold
   private static double avgError;         // calculated average error across test cases

/*
* Sets values for the parameters as declared and described above. These variables
* include those involved in configuring the structure of the network, training, calculating and applying
* weights, and reporting results and reasons for termination of the program.
*/
   public static void setConfigParams()
   {
      numIn = 2;
      numHidden = 20;
      numOut = 3;
      train = true;
      randomize = true;
      numTests = 4;
      errorThreshold = 0.0002;
      
      lambda = 0.3;
      avgError = 0.0;
      maxIters = 100000;
      minRand = 0.1;
      maxRand = 1.5;

      reasonToEndTrain = "";
      weightsFile = "networkWeights.txt";
   } //setConfigParams()

/*
* Prints particular configured parameters depending on whether the model is in training
* or running mode. Always prints the network configuration and which mode the model
* is operating in (running vs. training).
*/
   public static void echoConfigParams()
   {
      System.out.println("Network Configuration: " + numIn + "-" + numHidden + "-" + numOut);
      if (train)
      {
         System.out.println("The model is training.");
         System.out.println("Random number range (inclusive): [" + minRand + ", " + maxRand + "]");
         System.out.println("Max. Iterations/cycles: " + maxIters);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Lambda Value (learning factor): " + lambda);
      } // if (train)
      else
      {
         System.out.println("The model is running.");
      }
      System.out.println();
   } // echoConfigParams()

/*
* Utilizes the Math.random() method to generates a random value within the range of the given maximum and minimum.
* These values are used to randomize initial weights during training.
* 
* @return a randomly generated double within the given range
*/
   public static double generateRand()
   {
      return (double)((maxRand - minRand) * Math.random() + minRand);
   }

/*
* Calculates output when value is inputted into the sigmoid function.
*
* @param x the value which is used as the input to calculate the output of the sigmoid function
* @return the output of the sigmoid function when the input is passed through
*/
   public static double sigmoidFunction(double x)
   {
      return (1.0 / (1.0 + Math.exp(-x)));
   }

/*
* Calculates output when value is inputted into derivative of the sigmoid function.
*
* @param x the value which is used as the input to calculate the output of the derivative of the sigmoid function
* @return the output of the derivative of the sigmoid function when the input is passed through
*/
   public static double sigmoidDerivative(double x)
   {
      double y = sigmoidFunction(x);
      return (y * (1.0 - y));
   }  

/*
* Establishes the activation function, which is the mathematical basis for the 
* outputs of each node. In this case, our model uses a sigmoid activation function, which is written in the
* sigmoidFunction method.
* 
* @param x the value which is used as the input to calculate the output of the activation function
* @return the output of the activation function when the input is passed through
*/
   public static double activationFunction(double x)
   {
      return sigmoidFunction(x);
   }

/*
* Establishes the derivative of the activation function, which is the mathematical basis for the outputs 
* of each node. In this case, our model uses a derivative of the sigmoid activation function, which is written in mathematical
* terms in the sigmoidDerivative function.
* 
* @param x the value which is used as the input to calculate the output of the derivative of the activation function
* @return the output of the derivative of the activation function when the input is passed through
*/
   public static double activationFunctionPrime(double x)
   {
      return sigmoidDerivative(x);
   }

/*
* Designates memory for the arrays that were declared earlier and configured in the setConfigParams() method.
* This method will allocate memory of arrays only if necessary given the mode the program is in (training vs. running).
*/
   public static void allocateArrayMemory()
   {
      a = new double[numIn];
      h = new double[numHidden];
      wKJ = new double[numIn][numHidden];
      wJI = new double[numHidden][numOut];

      testingData = new double[numTests][numIn];
      testingExpected = new double[numTests][numOut];
      F = new double[numTests][numOut];
      errors = new double[numTests]; 
      
      if (train) // allocates memory for training-specific arrays
      {
         deltaWKJ = new double[numIn][numHidden];
         deltaWJI = new double[numHidden][numOut];

         errorWRTwKJ = new double[numIn][numHidden];
         errorWRTwJI = new double[numHidden][numOut];

         thetaJArray = new double[numHidden]; 
         thetaIArray = new double[numOut];
         psiJ = new double[numHidden];
         psiI = new double[numOut];
         omegaJ = new double[numHidden];
         omegaI = new double[numOut];
      } // if (train)
   } // allocateArrayMemory()

/*
* Fills the declared arrays after memory is allocated for them in the allocateMemory() method. 
* Arrays are populated based on the mode the program populating in (random vs. not). If the model is set
* to randomize, weights are randomly generated. Otherwise, weights are manually set. There is a manually set
* table of test cases and expected results.
*/
   public static void populateArrays()
   {
      if (randomize) // Populates weights using randomly generated numbers
      {
         for (k = 0; k < numIn; k++)
         {
            for (j = 0; j < numHidden; j++)
            {
               wKJ[k][j] = generateRand(); // Randomizes weights between input and hidden layers
            }
         }

         for (j = 0; j < numHidden; j++)
         {
            for (i = 0; i < numOut; i++)
            {
               wJI[j][i] = generateRand(); // Randomizes weights between hidden and output layers
            }
         }
      } // if (randomize)
      else // Populates weights in a non-random way, using manual input
      { 
/* 
* Sets weights manually, one at a time
*/
         for (k = 0; k < numIn; k++)
         {
            for (j = 0; j < numHidden; j++) 
            {
               wKJ[k][j] = 0.5;
            }
         }
         
         for (j = 0; j < numHidden; j++)
         {
            for (i = 0; i < numOut; i++)
            {
               wJI[j][i] = 0.5;
            }
         }
      } // else
/*
* Populates testing data and expected output values for the 4 cases
*/
      testingData[0][0] = 0.0;
      testingData[0][1] = 0.0;
      testingData[1][0] = 0.0;
      testingData[1][1] = 1.0;
      testingData[2][0] = 1.0;
      testingData[2][1] = 0.0;
      testingData[3][0] = 1.0;
      testingData[3][1] = 1.0;
    
      /* testingExpected[0][0] = 0.0;
      testingExpected[1][0] = 1.0;
      testingExpected[2][0] = 1.0;
      testingExpected[3][0] = 0.0; */

      testingExpected[0][0] = 0.0;
      testingExpected[0][1] = 0.0;
      testingExpected[0][2] = 0.0;

      testingExpected[1][0] = 0.0;
      testingExpected[1][1] = 1.0;
      testingExpected[1][2] = 1.0;

      testingExpected[2][0] = 0.0;
      testingExpected[2][1] = 1.0;
      testingExpected[2][2] = 1.0;

      testingExpected[3][0] = 1.0;
      testingExpected[3][1] = 1.0;
      testingExpected[3][2] = 0.0;
   } // populateArrays()

/*
* Trains the network until the maximum number of iterations has been reached or the
* average error is below the error threshold that was set before. For each training case,
* the model uses gradient (steepest) descent to minimize error by altering weights.
*/
   public static void train()
   {
      done = false;
      iterationsReached = 0;
      while (!done)  
      {
         for (t = 0; t < numTests; t++) 
         {
            for (k = 0; k < numIn; k++)  
            {
               a[k] = testingData[t][k]; // Sets input activations to values in the established dataset of test cases
            }
/* 
* Calculates theta values for the first layer and sets the hidden activation values
*/    
            for (j = 0; j < numHidden; j++)
            {
               thetaJArray[j] = 0.0;
               for (k = 0; k < numIn; k++) 
               {
                  thetaJArray[j] += a[k] * wKJ[k][j];
               }
               h[j] = activationFunction(thetaJArray[j]);
            }
/* 
* Calculates the theta, omega, and psi values for the second layer
*/ 
            for (i = 0; i < numOut; i++)
            {
               thetaIArray[i] = 0.0;
               for (j = 0; j < numHidden; j++) 
               {
                  thetaIArray[i] += h[j] * wJI[j][i];
               }
               F[t][i] = activationFunction(thetaIArray[i]);
            }
      
            for (i = 0; i < numOut; i++)
            {
               omegaI[i] = testingExpected[t][i] - F[t][i]; 
               psiI[i] = omegaI[i] * activationFunctionPrime(thetaIArray[i]);
            }
/*
* Calculates omega and psi values for the first layer
*/
            for (j = 0; j < numHidden; j++)    
            { 
               omegaJ[j] = 0.0;
               for (i = 0; i < numOut; i++)
               {        
                  omegaJ[j] += psiI[i] * wJI[j][i];
               }
            }

            for (j = 0; j < numHidden; j++)
            {
               psiJ[j] = omegaJ[j] * activationFunctionPrime(thetaJArray[j]);
            }
/*
* Calculates errors and changes in weights with respect to the weights between the first and second layers
*/
            for (k = 0; k < numIn; k++)
            {
               for (j = 0; j < numHidden; j++) 
               {
                  errorWRTwKJ[k][j] = -a[k] * psiJ[j];
                  deltaWKJ[k][j] = -lambda * errorWRTwKJ[k][j];
               }
            }    
/*
* Calculates errors and changes in weights with respect to the weights between the second and output layers
*/
            for (j = 0; j < numHidden; j++)
            {
               for (i = 0; i < numOut; i++)
               {
                  errorWRTwJI[j][i] = -h[j] * psiI[i];
                  deltaWJI[j][i] = -lambda * errorWRTwJI[j][i];
               }
            }

            applyWeights();
         }  // for (t = 0; t < numTests; t++)

         iterationsReached++;
         run();
/*
* Determines if progam should be terminated
*/
         if (iterationsReached >= maxIters || avgError <= errorThreshold)
         {
            done = true;
         }
      } // while (!done)
   } // train()

/*
* Saves/writes all of the network's weights into a file with the name specified in the variable weightsFile.
* First the weights between the input and hidden layers, then the weights between the hidden and output
* layers.
*/
   public static void saveWeights()
   {
      try (PrintWriter writer = new PrintWriter(new FileWriter(weightsFile)))
      {
/*
* Saves weights between input and hidden layers
*/
         for (k = 0; k < numIn; k++)
         {
            for (j = 0; j < numHidden; j++)
            {
               writer.print(wKJ[k][j] + " "); 
            }
            writer.println();
         }
         writer.println();
/*
* Saves weights between hidden and output layers
*/
         for (j = 0; j < numHidden; j++)
         {
            for (i = 0; i < numOut; i++)
            {
               writer.print(wJI[j][i] + " ");
            }
            writer.println();
         }
         writer.close();
      } // try (PrintWriter writer = new PrintWriter(new FileWriter(weightsFile)))
      catch (IOException e) 
      {
         System.out.println("IO exception caught writing to file");
      }
   } // saveWeights()

/*
* Reads the file (name specified in variable named weightsFile). Loads the weights
* from the file to build the network.
*/
   public static void loadWeights()
   {
      try (BufferedReader reader = new BufferedReader(new FileReader(weightsFile)))
      {
/*
* Loads weights between input and hidden layers
*/
         for (k = 0; k < numIn; k++)
         {
            String[] readIn = reader.readLine().trim().split("\\s+");
            for (j = 0; j < numHidden; j++)
            {
               wKJ[k][j] = Double.parseDouble(readIn[j]);
            }
         }
         reader.readLine();
/*
* Loads weights between hidden and output layers
*/
         for (j = 0; j < numHidden; j++)
         {
            String[] readIn = reader.readLine().trim().split("\\s+");
            for (i = 0; i < numOut; i++)
            {
               wJI[j][i] = Double.parseDouble(readIn[i]);
            }
         }
         reader.close();
      } // try (BufferedReader reader = new BufferedReader(new FileReader(weightsFile)))
      catch (IOException e) 
      {
         System.out.println("IO exception caught reading file");
      }
   } // loadWeights()

/*
* Applies the calculated changes in weights between hidden and ouput layers, then
* applies changes to weights between first and hidden layers.
*/
   public static void applyWeights()
   {
/*
* Applies changes in weights between hidden and output layers
*/
      for (j = 0; j < numHidden; j++)
      {
         for (i = 0; i < numOut; i++)
         {
            wJI[j][i] += deltaWJI[j][i];
         }
      }
/*
* Applies changes in weights between input and hidden layers
*/
      for (k = 0; k < numIn; k++)
      {
         for (j = 0; j < numHidden; j++) 
         {
            wKJ[k][j] += deltaWKJ[k][j];
         }
      }
   } // applyWeights()

/*
* Runs the model, calculating the model's output and errors for each given test case.
* The output is stored in F[][], while the errors are stored in errors[][]. Additionally,
* the average error is calculated.
*/
   public static void run()
   {
      for (t = 0; t < numTests; t++) 
      {
         for (k = 0; k < numIn; k++)  
         {
            a[k] = testingData[t][k]; // Sets input activations to values in the established dataset of test cases
         }
/* 
* Calculates theta value for the first layer and sets the hidden activation values
*/    
         for (j = 0; j < numHidden; j++)
         {
            thetaJ = 0.0;
            for (k = 0; k < numIn; k++) 
            {
               thetaJ += a[k] * wKJ[k][j];
            }
            h[j] = activationFunction(thetaJ);
         }
/* 
* Calculates the theta value for the second layer
*/ 
         errors[t] = 0.0;
         for (i = 0; i < numOut; i++)
         {
            thetaI = 0.0;
            for (j = 0; j < numHidden; j++) 
            {
               thetaI += h[j] * wJI[j][i];
            }
            F[t][i] = activationFunction(thetaI);
            omegaI[i] = testingExpected[t][i] - F[t][i];
            errors[t] += 0.5 * omegaI[i] * omegaI[i];
         } // for (i = 0; i < numOut; i++)
      } // for (t = 0; t < numTests; t++)
       
      avgError = 0.0;
      for (t = 0; t < numTests; t++)
      {
         avgError += errors[t];
      }
      avgError /= (double)numTests;
   } // run()

/*
* Prints values of parameters at the time when program was terminated. If model is running, 
* only the number of iterations and error reached printed. If the model is training, it will
* also print the program's reason for termination of training.
*/
   public static void reportResults()
   {
      if (iterationsReached >= maxIters)
      {
         reasonToEndTrain = "the max. number of iterations has been reached.";
      }
      if (avgError <= errorThreshold)
      {
         reasonToEndTrain = "the error threshold has been reached.";
      }
      if (train)
      {
         System.out.println("Reason for end of training: " + reasonToEndTrain);
      }
      System.out.println("Iterations reached: " + iterationsReached);
      System.out.println("Error reached: " + avgError);
   }  // reportResults()

/*
* Prints results in truth tables for easy reading of results. The truth table printed
* contains inputs, expected outputs from the given test cases, and values calculated using the model.
* The data are printed and evaluated based on the test cases set in populateArrays().
*/
   public static void reportTruthTables()
   {
      System.out.println();
      if (!train) // determines which mode (running vs. training) the model is operating in, and prints info accordingly
      {
         System.out.println("Results for recent run");
      } 
      else
      {
         System.out.println("Results for recent training");
      }
      System.out.println("(IN. = input, EXP. = expected, OUT. = output)"); // meanings of abbreviated titles in table header
/**
* Prints truth table with inputs, expected outputs, and calculated outputs.
*/
      for (k = 0; k < numIn; k++)
      {
         System.out.print(" IN. " + (k + 1) + "  |");
      }
      for (i = 0; i < numOut; i++)
      {
         System.out.print("  EXP. " + (i + 1) + "  |");
      }
      for (i = 0; i < numOut; i++)
      {
         System.out.print("   OUT. " + (i + 1) + "   |");
      }
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------------------");
      for (t = 0; t < numTests; t++)
      {
         System.out.print("   ");
         for (k = 0; k < numIn; k++)
         {
            System.out.print(testingData[t][k] + "  |   ");
         }
         for (i = 0; i < numOut; i++)
         {
            System.out.print(testingExpected[t][i] + "    |   ");
         }
         for (i = 0; i < numOut; i++)
         {
            System.out.printf("%.7f |  ", F[t][i]);
         }
         System.out.println();
      } // for (t = 0; t < numTests; t++)
      saveWeights();
   } // reportTruthTables()

/*
* Tests the perceptron by using all the methods necessary for training and running, depending on
* which mode the model is operating in.
*/
   public static void main(String[] args)
   {
      setConfigParams();
      echoConfigParams();

      allocateArrayMemory();
      populateArrays();

      if (train)
      {
         train();
      }
      else 
      {
         run();
      }
      reportResults();
      reportTruthTables();
   } // main(String[] args)
} // public class Perceptron