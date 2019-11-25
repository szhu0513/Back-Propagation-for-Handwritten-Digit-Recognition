import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        // TODO: add code here
    	double softmax = 0;
    	int idx = 0;
    	
    	for(int x = 0; x < inputNodes.size()-1; x++) {
    		double input = instance.attributes.get(x);
    		inputNodes.get(x).setInput(input);
    	}   
    	
        for(int x = 0; x < hiddenNodes.size(); x++) {
        	 hiddenNodes.get(x).calculateOutput();
        }  
        
        for(int x = 0; x < outputNodes.size(); x++) {
            outputNodes.get(x).calculateOutput();
            softmax +=  outputNodes.get(x).getOutput();
        }
        
        for (int x = 0; x < outputNodes.size(); x++) {
        	outputNodes.get(x).getSoftmax(softmax);
        }
            
        double highestValue = outputNodes.get(0).getOutput();
        for(int x = 1; x < outputNodes.size(); x++) {
            if (outputNodes.get(x).getOutput( )> highestValue) {
            	highestValue = outputNodes.get(x).getOutput();
            	idx = x;
            }
        }
        
        return idx;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        // TODO: add code here
    	for (int epoch = 0; epoch < maxEpoch; epoch++) {
    		double sumOfLoss = 0.0;
    		Collections.shuffle(trainingSet,random);
    		
    		for (int x = 0; x < trainingSet.size(); x++ ) {
    			predict(trainingSet.get(x));
    			
    			for (int y = 0; y < outputNodes.size(); y++) {
                    outputNodes.get(y).targetClassValue = (trainingSet.get(x).classValues).get(y);
                    outputNodes.get(y).calculateDelta();
                }
    			
    			for(int y = 0; y < hiddenNodes.size(); y++) {
                    double sum = 0.0;
                    for(int z = 0; z < outputNodes.size(); z++) {
                    	double w = outputNodes.get(z).parents.get(y).weight;
                    	double d = outputNodes.get(z).delta;
                        sum += w*d;
                    }
                    hiddenNodes.get(y).targetClassValue = sum;
                    hiddenNodes.get(y).calculateDelta();
                }
    			
    			for(int y = 0; y < hiddenNodes.size(); y++) {
                	hiddenNodes.get(y).updateWeight(learningRate);
                }
    			
    			for(int y = 0; y < outputNodes.size(); y++) {
    				outputNodes.get(y).updateWeight(learningRate);
    			}

    		}
    		
    		for (int x = 0; x < trainingSet.size(); x++ ) {
    			sumOfLoss += loss(trainingSet.get(x));
    		}
    		
    		sumOfLoss = sumOfLoss / trainingSet.size();   		
    		String result = String.format("%.3e",sumOfLoss);
            System.out.println("Epoch: " + epoch + ", Loss: " + result);
    	}
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here
    	double loss = 0.0;
    	predict(instance);
        for(int i = 0; i < instance.classValues.size(); i++) {
            loss += Math.log(outputNodes.get(i).getOutput())*instance.classValues.get(i);
        }
        return -loss;
    }
}
