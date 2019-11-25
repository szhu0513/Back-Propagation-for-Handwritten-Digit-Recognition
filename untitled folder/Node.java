import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    public double delta = 0.0; //input gradient
    public double targetClassValue =0.0;

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    //The implementation of ReLU activation function
    private double getReLU(double in) {
        return Math.max(in, 0.0);
    }
    
    //The implementation of the derivative of ReLU function
    private double getDerivativeOfReLU(double in) {
    	if (in <= 0) {
            return 0.0;
        } else {
            return 1.0;
        }
    }
    
    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        double sum = 0.0;
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here
            if (type == 2) {
                //use ReLU activation function
                for (int x = 0; x < this.parents.size(); x++) {
                    double xi2 = this.parents.get(x).node.getOutput();
                    double wi2 = this.parents.get(x).weight;
                    sum += xi2 * wi2;
                }
                this.outputValue = getReLU(sum);
                this.outputGradient=getDerivativeOfReLU(sum);
            }
            if (type == 4) {
                //use Softmax activation function 
                for (int x = 0; x < this.parents.size(); x++) {
                    double xi4 = this.parents.get(x).node.getOutput();
                    double wi4 = this.parents.get(x).weight;
                    sum += xi4 * wi4;
                }
                this.outputValue = Math.exp(sum);
            }
        }
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
    public void calculateDelta() {
        if (type == 2 || type == 4)  {
            // TODO: add code here
            if (type == 2) {
                this.delta = this.targetClassValue * this.outputGradient;
            }
            if (type == 4) {
                this.delta=this.targetClassValue - this.getOutput();
            }
        }
    }


    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here
        	for (int x = 0; x < this.parents.size(); x++) {
            	double ai = this.parents.get(x).node.getOutput();
                this.parents.get(x).weight += learningRate*ai*this.delta;
            } 
        }
    }
    
    public void getSoftmax(double in){
        this.outputValue = this.getOutput()/in;
    }
}


