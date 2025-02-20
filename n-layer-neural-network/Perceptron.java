import java.util.*;
import java.io.*;

/*
* @author Emily Kwan
* @version 5/7/24
* 
* The Perceptron class contains code for an N-layer network. 
* This neural network trains and alters weights using a gradient (steepest) descent method in order to minimize error, 
* and it also contains code to perform the backpropagation algorithm. This network applies a sigmoid
* activation function to calculate outputs. This class also includes functionalities to externally configure the model 
* and print different results based on the mode it is operating in, whether that is training or running. The 
* model's test cases and expected outputs are manually set.
*
* Table of Contents:
*     public static void setConfigParams()
*     public static void echoConfigParams()
*     public static double generateRand()
*     public static double sigmoidFunction(double x)
*     public static double sigmoidDerivative(double x)
*     public static double activationFunction(double x)
*     public static double activationFunctionPrime(double x)
*     public static void allocateArrayMemory()
*     public static int findMaxDimension()
*     public static void populateArrays()
*     public static void saveWeights()
*     public static void loadWeights()
*     public static void setUp(int t1)
*     public static void train()
*     public static void run()
*     public static void runForTrain(int t1)
*     public static void runCurrentNetwork()
*     public static void reportResults()
*     public static void reportTruthTables()
*     public static void printWeights()
*     public static void printActivations()
*     public static void main(String[] args)
*/

public class Perceptron
{
/* 
* Declaration of network configuration variables  
*/ 
   private static int numLayers;             // number of connectivity layers in network
   private static int numActivationLayers;   // number of activation layers in network
   private static int numIn;                 // number of input activations
   private static int numOut;                // number of output activations
   private static boolean train;             // identifies whether or not the model will be trained; if false, it is just ran
   private static boolean randomize;         // identifies if weights will be randomized; if false, they are manually set
   private static boolean load;              // identifies if weights will be loaded
   private static boolean save;              // identifies if weights will be saved
   private static double errorThreshold;     // the program will terminate if the error is below this threshold
   private static int numTests;              // number of test cases

/*
* Declaration of arrays of activations, weights, final output, and activations per layer
*/
   private static double[][] a;              // array of activations
   private static double[][][] w;            // array of network weights
   private static double[][] networkOutput;  // array of ultimate network output that is reported
   private static int[] activationsPerLayer; // array that stores number of activations in each layer

/*
 * Names of files that parameters are loaded from/saved to
 */
   private static String weightsFile;        // name of the file that contains saved/loaded weights
   private static String configFile;         // name of the file that contains network configuration parameters
   private static String testingDataFile;    // name of the file that contains testing data input activation values

/* 
* Declaration of arrays and tables with testing inputs/expected outputs
*/
   private static double[][] testingData;     // table with manually initialized values with test cases for running and training
   private static double[][] testingExpected; // expected calculated values using the given test cases

/*
* Declaration of values and arrays used in training and calculating weights
*/
   private static double lambda;          // learning factor--controls how quickly weights are changed
   private static double minRand;         // minimum value that randomly generated weights can be (inclusive)
   private static double maxRand;         // maximum value that randomly generated weight can be (inclusive)
   private static int maxIters;           // maximum number of iterations the program can reach
   private static boolean done;           // identifies whether or not the program is done training or not
   private static double[][] thetaArray;  // array of saved theta values
   private static double[][] psiArray;    // array of saved psi values

/*
* Declaration of variables used to report results
*/
   private static int iterationsReached;   // amount of iterations the program has reacher
   private static String reasonToEndTrain; // reason for training end: either max. iter. reached, or avg. error is below threshold
   private static double avgError;         // calculated average error across test cases
   private static double trainingTime;     // amount of time network took to train/run
   private static double modIterations;    // how many iterations before another keep alive message is printed


/*
* Sets values from config file for the parameters as declared and described above. These variables
* include those involved in configuring the structure of the network, training, calculating and applying
* weights, and reporting results and reasons for termination of the program.
*/
   public static void setConfigParams()
   {
      try (BufferedReader br = new BufferedReader(new FileReader(configFile)))
      {
         br.readLine();
         br.readLine();
         br.readLine();

/*
 * Reads in parameters that define the configuration of the network and test cases
 */
         numLayers = Integer.parseInt(br.readLine().split("= ")[1]);

         String[] readInActivationsPerLayer = br.readLine().split("= ")[1].split("-");

         numActivationLayers = readInActivationsPerLayer.length;

         activationsPerLayer = new int[numActivationLayers];

         for (int n = 0; n < numActivationLayers; n++)
         {
            activationsPerLayer[n] = Integer.parseInt(readInActivationsPerLayer[n]);
         }
         
         numIn = activationsPerLayer[0];
         numOut = activationsPerLayer[numLayers];
         numTests = Integer.parseInt(br.readLine().split("= ")[1]);
         br.readLine();
         br.readLine();

/*
 * Reads in booleans that determine how network is trained/weighted
 */
         train = br.readLine().split("= ")[1].equals("1");
         randomize = br.readLine().split("= ")[1].equals("1");
         load = br.readLine().split("= ")[1].equals("1");
         save = br.readLine().split("= ")[1].equals("1");
         br.readLine();
         br.readLine();

/*
 * Reads in bounds for the values of randomized weights
 */
         minRand = Double.parseDouble(br.readLine().split("= ")[1]);
         maxRand = Double.parseDouble(br.readLine().split("= ")[1]);
         br.readLine();
         br.readLine();

/*
 * Reads in values used in training
 */
         maxIters = Integer.parseInt(br.readLine().split("= ")[1]);
         errorThreshold = Double.parseDouble(br.readLine().split("= ")[1]);
         lambda = Double.parseDouble(br.readLine().split("= ")[1]);
         modIterations = Integer.parseInt(br.readLine().split("= ")[1]);
         br.readLine();
         br.readLine();

/*
 * Reads in names of files for weights and testing data activation values
 */
         weightsFile = br.readLine().split(": ")[1];   
         testingDataFile = br.readLine().split(": ")[1]; 

         br.close();
      } // try (BufferedReader br = new BufferedReader(new FileReader(configFile)))
      catch (IOException e)
      {
         System.out.println("IO exception caught reading file");
      }

      reasonToEndTrain = "";
   } //setConfigParams()

/*
* Prints particular configured parameters depending on whether the model is in training
* or running mode. Always prints the network configuration and which mode the model
* is operating in (running vs. training).
*/
   public static void echoConfigParams()
   {
      System.out.print("\nNetwork Configuration: ");

      for (int n = 0; n < numActivationLayers - 1; n++)
      {
         System.out.print(activationsPerLayer[n] + "-");
      }

      System.out.println(numOut);
      System.out.println();

      if (train)
      {
         System.out.println("The model is training.");
         System.out.println("Random number range (inclusive): [" + minRand + ", " + maxRand + "]");
         System.out.println("Max. Iterations/cycles: " + maxIters);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Lambda Value (learning factor): " + lambda);
         System.out.println("Weights are loaded/save from/to " + weightsFile);
         System.out.println("Testing data activation values are loaded from " + testingDataFile);
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
      return ((maxRand - minRand) * Math.random() + minRand);
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
      int maxDimension = findMaxDimension();

      a = new double[numActivationLayers][maxDimension];
      
      w = new double[numActivationLayers][maxDimension][findSecondMaxDimension()];

      testingData = new double[numTests][numIn];
      testingExpected = new double[numTests][numOut];

      networkOutput = new double[numTests][numOut];
      
      if (train) // allocates memory for training-specific arrays
      {
         thetaArray = new double[numActivationLayers][maxDimension];
         psiArray = new double[numActivationLayers][maxDimension];
      } // if (train)
   } // allocateArrayMemory()

/**
 * Finds maximum dimension by comparing amounts of nodes in each layer
 * @return the maximum amount of nodes across layers
 */
   public static int findMaxDimension()
   {
      int max = numIn;

      for (int n = 1; n < numActivationLayers; n++)
      {
         if (activationsPerLayer[n] > max)
         {
            max = activationsPerLayer[n];
         }
      }

      return max;
   } // findMaxDimension()

   public static int findSecondMaxDimension()
   {
      int max = 0;

      for (int n = 1; n < numActivationLayers; n++)
      {
         if (activationsPerLayer[n] > max)
         {
            max = activationsPerLayer[n];
         }
      }

      return max;
   } // findMaxDimension()

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
         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < activationsPerLayer[n]; k++)
            {
               for (int j = 0; j < activationsPerLayer[n + 1]; j++)
               {
                  w[n][k][j] = generateRand();
               }
            }
         } // for (int n = 0; n < numLayers; n++)
      } // if (randomize)
      else if (load) // Checks if weights should be loaded from file
      { 
         loadWeights();
         System.out.println("weights were loaded from " + weightsFile);
      }
      else // Populates weights in a non-random way, using manual input
      {
         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < activationsPerLayer[n]; k++)
            {
               for (int j = 0; j < activationsPerLayer[n + 1]; j++)
               {
                  w[n][k][j] = 0.5;
               }
            }
         } // for (int n = 0; n < numLayers; n++)
      } // else

/*
* Reads in and populates testing data from testing data activations file
*/
      try (BufferedReader bReader = new BufferedReader(new FileReader(testingDataFile)))
      {
         bReader.readLine();
         bReader.readLine();
         bReader.readLine();

         for (int t = 0; t < numTests; t++)
         {
            String line = bReader.readLine();
            if (line != null)
            {
               for (int m = 0; m < numIn; m++)
               {
                  testingData[t][m] = Double.parseDouble(line.split(" ")[m]);
                  // DEBUG: System.out.println("testingData[" + t + "][" + m + "] = " + testingData[t][m]);
               }
            }
         } // for (int t = 0; t < numTests; t++)

         bReader.readLine();
         bReader.readLine();

         bReader.close();
      } // try (BufferedReader bReader = new BufferedReader(new FileReader(configFile)))
      catch (IOException e)
      {
         System.out.println("IO exception caught reading file");
      }

/*
* Reads in and populates expected testing output values from config file
*/
      try (BufferedReader b = new BufferedReader(new FileReader(configFile)))
      {

/*
 * Skips top of config file with config params that have already been intialized
 */
         String top = b.readLine();

         while (!top.equals("// Test case expected outputs"))
         {
            top = b.readLine();
         }

/*
 * Reads in testing expected output data from bottom of config file.
 */
         for (int t = 0; t < numTests; t++)
         {
            String line = b.readLine();

            if (line != null)
            {
               for (int i = 0; i < numOut; i++)
               {
                  testingExpected[t][i] = Double.parseDouble(line.split(" ")[i]);
                  // DEBUG: System.out.println("testingExpected[" + t + "][" + i + "] = " + testingExpected[t][i]);
               }
            }
         } // for (int t = 0; t < numTests; t++)

         b.close();
      } // try (BufferedReader b = new BufferedReader(new FileReader(configFile)))
      catch (IOException e)
      {
         System.out.println("IO exception caught reading file");
      }
   } // populateArrays()

/*
* Saves/writes all of the network's weights into a file with the name specified in the variable weightsFile.
* First the weights between the input and hidden layers, then the weights between the hidden and output
* layers.
*/
   public static void saveWeights()
   {
      try (PrintWriter writer = new PrintWriter(new FileWriter(weightsFile)))
      {
         writer.print("These are the weights for a " + numLayers + " layer network with a "); 

         for (int n = 0; n < numActivationLayers - 1; n++)
         {
            writer.print(activationsPerLayer[n] + "-");
         }

         writer.print(numOut + " configuration.");
         writer.println();
         writer.println();

/*
* Traverses and writes weights to specified file
*/
         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < activationsPerLayer[n]; k++)
            {
               for (int j = 0; j < activationsPerLayer[n + 1]; j++)
               {
                  writer.print(w[n][k][j] + " ");
               }

               writer.println();
            }

            writer.println();
         } // for (int n = 0; n < numLayers; n++)

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
         reader.readLine();
         reader.readLine();

/*
* Traverses and loads weights from specified file
*/

         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < activationsPerLayer[n]; k++)
            {
               String[] readIn = reader.readLine().trim().split("\\s+");

               for (int j = 0; j < activationsPerLayer[n + 1]; j++)
               {
                  w[n][k][j] = Double.parseDouble(readIn[j]);
               }
            }

            reader.readLine();
         } // for (int n = 0; n < numLayers; n++)

         reader.close();
      } // try (BufferedReader reader = new BufferedReader(new FileReader(weightsFile)))
      catch (IOException e) 
      {
         System.out.println("IO exception caught reading file");
      }
   } // loadWeights()
   
/*
 * Loads input activations into array of activations from testing dataset
 * 
 * @param t the test case that we are getting the activation values from
 */
   public static void setUp(int t1)
   {
      for (int m = 0; m < numIn; m++)  
      {
         a[0][m] = testingData[t1][m]; // Sets input activations to values in the established dataset of test cases
         // DEBUG: System.out.println("a[0][" + m + "] = testingData[" + t1 + "][" + m + "] = " + testingData[t1][m]); 
      }
   } // setUp()
   
/*
* Trains the network until the maximum number of iterations has been reached or the
* average error is below the error threshold that was set before. For each training case,
* the model uses gradient (steepest) descent and the backpropagation algorithm
* to minimize error by altering weights.
*/
   public static void train()
   {
      done = false;
      iterationsReached = 0;

      while (!done)  
      {
         double totalError = 0.0;
         
         for (int t = 0; t < numTests; t++) 
         {
            setUp(t); // load in testing inputs

            runForTrain(t);

/*
* Calculates omega values and weights to implement the backpropagation algorithm.
*/

            double omegaVal;

            for (int n = numLayers - 1; n > 1; n--)
            {
               for (int j = 0; j < activationsPerLayer[n]; j++)
               {
                  omegaVal = 0.0;

                  for (int k = 0; k < activationsPerLayer[n + 1]; k++)
                  {
                     omegaVal += psiArray[n + 1][k] * w[n][j][k];
                     w[n][j][k] += lambda * a[n][j] * psiArray[n + 1][k];
                  }

                  psiArray[n][j] = omegaVal * activationFunctionPrime(thetaArray[n][j]);
               } // for (int j = 0; j < activationsPerLayer[n]; j++)
            } // for (int n = numLayers - 1; n > 1; n--)
            
/*
 * Calculates values for final layer of network
 */
            int n = 1;

            for (int j = 0; j < activationsPerLayer[n]; j++)
            {
               omegaVal = 0.0;

               for (int k = 0; k < activationsPerLayer[n + 1]; k++) 
               {
                  omegaVal += psiArray[n + 1][k] * w[n][j][k];
                  w[n][j][k] += lambda * a[n][j] * psiArray[n + 1][k];
               }

               psiArray[n][j] = omegaVal * activationFunctionPrime(thetaArray[n][j]);

               for (int m = 0; m < numIn; m++)
               {
                  w[n - 1][m][j] += lambda * a[n - 1][m] * psiArray[n][j];
               }
            } // for (int j = 0; j < activationsPerLayer[n]; j++)

            runCurrentNetwork();
            
/*
 * Accumulates/calculates total error
 */
            double omega;

            for (int i = 0; i < numOut; i++)
            {
               omega = testingExpected[t][i] - a[numLayers][i];
               totalError += 0.5 * omega * omega;
            }

         } // for (int t = 0; t < numTests; t++)

         iterationsReached++;

         avgError = totalError / (double) numTests;

/*
* Determines if progam should be terminated
*/
         done = iterationsReached >= maxIters || avgError <= errorThreshold;

/*
 * Outputs messages with iteration count and average error
 */
         if (iterationsReached % modIterations == 0)
         {
            System.out.printf("Iteration %d, Average Error = %f\n", iterationsReached, avgError);
         }

      } // while (!done)
      System.out.println();
   } // train()

/*
* Runs the model, calculating the model's output for each given test case.
* The output is ultimately stored in networkOutput[][].
*/
   public static void run()
   {
      for (int t = 0; t < numTests; t++) 
      {
         setUp(t);

         runCurrentNetwork();

         for (int i = 0; i < numOut; i++)
         {
            networkOutput[t][i] = a[numLayers][i];
         }
      } // for (int t = 0; t < numTests; t++) 
   } // run()

/*
* Runs the model, calculating and saving theta values.
*/
   public static void runForTrain(int t1)
   {
/*
 * Calculates activations, theta, and psi values, for layers until numLayers - 1
 */
      for (int n = 1; n < numLayers; n++)
      {
         // DEBUG: System.out.println("activationsPerLayer[" + n + "] = " + activationsPerLayer[n]);
         for (int j = 0; j < activationsPerLayer[n]; j++)
         {
            thetaArray[n][j] = 0.0;

            for (int k = 0; k < activationsPerLayer[n - 1]; k++)
            {
               thetaArray[n][j] += a[n - 1][k] * w[n - 1][k][j];
            }
            
            a[n][j] = activationFunction(thetaArray[n][j]);
            // DEBUG: System.out.println("theta[" + n + "][" + j + "] = " + thetaArray[n][j]);
         } // for (int j = 0; j < activationsPerLayer[n]; j++)
      } // for (int n = 1; n < numLayers; n++)

/* 
* Calculates the activations, theta, and psi values for output layer
*/ 
      int n = numLayers;

      for (int i = 0; i < numOut; i++)
      {
         thetaArray[n][i] = 0.0;

         for (int j = 0; j < activationsPerLayer[n - 1]; j++) 
         {
            thetaArray[n][i] += a[n - 1][j] * w[n - 1][j][i];
         }

         a[n][i] = activationFunction(thetaArray[n][i]);
         psiArray[n][i] = (testingExpected[t1][i] - a[n][i]) * activationFunctionPrime(thetaArray[n][i]);
      } // for (int i = 0; i < numOut; i++)
   } // runForTrain(int t1)

/*
 * Runs the network and calculates theta values, but doesn't save them.
 */
   public static void runCurrentNetwork()
   {
/* 
* Calculates theta values for all layers
*/   
      double thetaVal; 

      for (int n = 1; n < numActivationLayers; n++)
      {
         for (int j = 0; j < activationsPerLayer[n]; j++)
         {
            thetaVal = 0.0;

            for (int k = 0; k < activationsPerLayer[n - 1]; k++)
            {
               thetaVal += a[n - 1][k] * w[n - 1][k][j];
            }

            a[n][j] = activationFunction(thetaVal);
            // DEBUG: System.out.println("a[" + (n) + "][" +j +  "] = " + a[n][j]);

         } // for (int j = 0; j < activationsPerLayer[n]; j++)
      } // for (int n = 1; n < numActivationLayers; n++)
   } // runCurrentNetwork()

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
      System.out.println("Network trained for: " + trainingTime + " milliseconds");
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

/*
* Prints truth table with inputs, expected outputs, and calculated outputs.
*/
      /* for (int m = 0; m < numIn; m++)
      {
         System.out.print(" IN. " + (m + 1) + "  |");
      } */

      /* for (int i = 0; i < numOut; i++)
      {
         System.out.print("  EXP. " + (i + 1) + "  |");
      } */

      /* for (int i = 0; i < numOut; i++)
      {
         System.out.print("         OUT. " + (i + 1) + "       |");
      } */

      /* System.out.println();
      System.out.println("--------------------------------------------------------------------------------------" +
                           "---------------------------------"); */

      for (int t = 0; t < numTests; t++)
      {
         /* System.out.print("   ");

         for (int m = 0; m < numIn; m++)
         {
            System.out.print(testingData[t][m] + "  |   ");
         }

         for (int i = 0; i < numOut; i++)
         {
            System.out.print(testingExpected[t][i] + "    |   ");
         } */

         for (int i = 0; i < numOut; i++)
         {
            System.out.printf("%.10f  ", networkOutput[t][i]);
         }
   
         System.out.println();
      } // for (int t = 0; t < numTests; t++)
   } // reportTruthTables()

/*
 * Prints the weights layer by layer
 */
   public static void printWeights()
   {
      for (int n = 0; n < numLayers; n++)
      {
         System.out.println("layer " + n + " of weights\n");

         for (int k = 0; k < activationsPerLayer[n]; k++)
         {
            for (int j = 0; j < activationsPerLayer[n + 1]; j++)
            {
               System.out.print(w[n][k][j] + " ");
            }

            System.out.println();
         }

         System.out.println();
      } // for (int n = 0; n < numLayers; n++)

      System.out.println("\n");
   } // printWeights()

/*
 * Prints activations layer by layer
 */
   public static void printActivations()
   {
      for (int n = 0; n < numActivationLayers; n++)
      {
         System.out.println("layer " + n + " of activations\n");

         for (int k = 0; k < activationsPerLayer[n]; k++)
         {
            System.out.print(a[n][k] + " ");
         }

         System.out.println("\n");
      } // for (int n = 0; n < numActivationLayers; n++)

      System.out.println("\n");
   } // printActivations()

/*
* Tests the perceptron by using all the methods necessary for training and running, depending on
* which mode the model is operating in. Also calculates time it took to run model.
*/
   public static void main(String[] args)
   {
      long startTime = System.currentTimeMillis(); // set start time for network running

/*
 * Set config file either to the default file, or a file defined in the command line
 */
      if (args.length > 0)
      {
         configFile = args[0];
      }
      else
      {
         configFile = "nlayer_config.txt";
      }

      setConfigParams();
      echoConfigParams();

      allocateArrayMemory();

      populateArrays();

      if (train)
      {
         train();
      }
      run();
      
/*
* Set end time and determine time that network took to run.
*/
      long endTime = System.currentTimeMillis();
      trainingTime = (double) (endTime - startTime);

      reportResults();
      reportTruthTables();

      if (save)
      {
         saveWeights();
         System.out.println("\nWeights have been saved to " + weightsFile + "\n");
      }
   } // main(String[] args)
} // public class Perceptron