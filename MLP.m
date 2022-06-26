

classdef MLP < handle
    % Member data
    properties (SetAccess=private)
        inputDimension % Number of inputs
        hiddenDimension % Number of hidden neurons
        outputDimension % Number of outputs
        
        hiddenLayerWeights % Weight matrix for the hidden layer, format (hiddenDim)x(inputDim+1) to include bias terms
        outputLayerWeights 
        count;
 
        mse;
        x2;
        x3;
 
        % Weight matrix for the output layer, format (outputDim)x(hiddenDim+1) to include bias terms
       
    end
    
    methods
        function mlp=MLP(inputD,hiddenD,outputD)
            mlp.inputDimension=inputD;
            mlp.hiddenDimension=hiddenD;
            mlp.outputDimension=outputD;
            mlp.hiddenLayerWeights=zeros(hiddenD,inputD+1);
            mlp.outputLayerWeights=zeros(outputD,hiddenD+1);
            mlp.x2;
            mlp.count;
            mlp.mse;
            mlp.x3;
          
        end
        function mlp=initializeWeightsRandomly(mlp,stdDev)
            % Note: 'mlp' here takes the role of 'this' (Java/C++) or
            % 'self' (Python), refering to the object instance this member
            % function is run on.
            mlp.hiddenLayerWeights = zeros(mlp.hiddenDimension,mlp.inputDimension + 1);
            mlp.outputLayerWeights = zeros(mlp.outputDimension,mlp.hiddenDimension +1);
            
            mlp.hiddenLayerWeights=rand(mlp.hiddenDimension,mlp.inputDimension + 1)*0.007+stdDev;
            mlp.outputLayerWeights=rand(mlp.outputDimension,mlp.hiddenDimension +1)*0.007+stdDev;
       
            mlp.count = 0;
            mlp.mse;
            
            
        end
        function [hidden,output]=compute_forward_activation(mlp, inputData)
               W1 = mlp.hiddenLayerWeights;
               W2 = mlp.outputLayerWeights; 
               v1 = inputData; 
               v1(end+1) = 1; 
             
               H = W1 * v1;
               for i = 1:mlp.hiddenDimension        
                    mlp.x2(i) = exp(H(i)) / (exp(H(i)) + 1);
               end

               v2 = mlp.x2;
               v2(end +1) = 1; 

               x = W2 * v2';
               mlp.x3 = x;
               hidden = H;

               for i = 1:mlp.outputDimension
                    out(i) = 1/(1+exp(-x(i)));   %sigmoid for output layer
               end
               output = out;      
        end
        function output=compute_output(mlp,input)
            [~,output] = mlp.compute_forward_activation(input);
            
        end
        function mlp=train_single_data(mlp, inputData, targetOutputData, learningRate)
            [h,o] = mlp.compute_forward_activation(inputData);
           
            W1 = mlp.hiddenLayerWeights;
            W2 = mlp.outputLayerWeights;

            y = targetOutputData;%target output
            v2 = mlp.x2; %assigning to sigmoid of hidden neurons
            v2(end+1) = 1; %assigning bias term
            v1 = inputData; %assigning all input data to v1
            v1(end+1) = 1; %assigning bias term
            
            k = find(y==1);
            
            loss = y(k) - o(k);
            disp("CONFIDENCE: ")
            disp(o(k));
            disp("ACTUAL: ")
            disp(y(k));
            disp(o);
            mlp.mse(end + 1) = loss;

            d = (o - transpose(y)) .* (o .* (1 - o)); %chain-rule, d = partial-derivative of cost fucntion in respect to sigmoid(H)
            d2 = (W2(:,1:mlp.hiddenDimension) .* d') .*mlp.x2.*(1 -mlp.x2);
            dW2 = d.*v2';
            for i = 1:mlp.hiddenDimension
                q(i) = sum(d2(:,i));
            end
            dW1 = q.*v1;
            
            W1 = W1- learningRate .* dW1'; %new Weights = oldWeights - lR*weightShifts
            W2 = W2 -learningRate .* dW2'; 
            mlp.hiddenLayerWeights = W1;
            mlp.outputLayerWeights = W2;    

        end
function ms = getMSQ(mlp)
    ms = mlp.mse;
end   


    end
    
        
       
end


  

