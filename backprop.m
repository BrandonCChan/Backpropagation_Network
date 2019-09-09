%Brandon Chan 10097604 10/16/2016
classdef backprop < handle
    %backprop class: Defines a class with attributes and functions to train
    %and store a back propagation network.
    %internal functions include a cost function that calculates mean square
    %error, sigmoid functions, and a classifier function to test a trained
    %instance of the network.
    %
    %Algorithms and implentation was based of of formulas outlined in
    %notes provided in the course CISC 452 and a series of instuctional
    %videos made by the Welch Labs youtube channel
    
    properties
        weights_input2hidden
        weights_hidden2output
    end
    
    properties (Access = private)
        num_attributes
        num_input_entries
        num_output_entries
        num_hidden_nodes
        num_output_nodes
        iteration_limit
        learning_rate
        momentum
    end
    
    methods
        %Object constructor that initialized various attributes based on
        %input variables, randomized numbers, and hard coded valued
        function obj = backprop(a, i, o, h, o_n)
            obj.num_attributes = a+1; %+1 to account for bias
            obj.num_input_entries = i;
            obj.num_output_entries = o;
            obj.num_hidden_nodes = h;
            obj.num_output_nodes = o_n;
            
            %Initalizes weights with random numbers between -1 and 1
            obj.weights_input2hidden = -1 + (1+1)*rand(obj.num_attributes,h); 
            obj.weights_hidden2output = -1 + (1+1)*rand(h,obj.num_output_nodes); 
            
            %Manually adjustable varibles
            obj.iteration_limit = 500;
            obj.learning_rate = 0.1;
            obj.momentum = 1;
        end
        
        %Trains the network using the back prop algorithm in batch
        %processing
        function mse = train_network(obj, inputs, outputs)
            bias_vec = ones(obj.num_input_entries,1);
            input_data = horzcat(inputs,bias_vec);
            output_data = outputs;
            
            %Map the desired outputs to a 10 node output layer format
            desired_outputs = zeros(obj.num_input_entries, obj.num_output_nodes);
            for i = 1:obj.num_input_entries
                index = output_data(i) + 1;
                desired_outputs(i,index) = 1;
            end
                      
            iteration = 1; %starting the iteration counter at 1
            
            predicted_outputs = rand(obj.num_input_entries,obj.num_output_nodes);
            
            %Calculate initial mse 
            mse = obj.calculate_mse(predicted_outputs, desired_outputs);
            disp('Initial MSE:');
            disp(mse);
            
            %While loop to facilitate network training
            while (mse > 0.02) && (iteration < obj.iteration_limit)
                
                for entry = 1:obj.num_input_entries
                    %Step 1: Compute hidden node inputs: Input * Weights_1 
                    activation_input2hidden = input_data(entry,:) * obj.weights_input2hidden; 

                    %Step 2: Compute hidden node output: sigmoided activation
                    activity_hidden = activation_input2hidden; 
                    activation_input2hidden_prime = activation_input2hidden;
                    for i = 1:obj.num_hidden_nodes
                        activity_hidden(i) = obj.sigmoid(activity_hidden(i));
                        activation_input2hidden_prime(i) = obj.sigmoid_prime(activation_input2hidden_prime(i));
                    end
                    
                    %Step 3: Compute output node inputs: Activity hidden layer * Weights_2
                    activation_hidden2output = activity_hidden * obj.weights_hidden2output; 
                   
                    %Step 4: Calculate output/prediced output
                    predicted_output = activation_hidden2output; 
                    activation_hidden2output_prime = activation_hidden2output;
                    for i = 1:obj.num_output_nodes
                        predicted_output(i) = obj.sigmoid(predicted_output(i));
                        activation_hidden2output_prime(i) = obj.sigmoid_prime(activation_hidden2output_prime(i));
                    end
                    
                    predicted_outputs(entry,:) = predicted_output; %Update "complete" output array

                    %Step 5: back propagation: update weights
                    delta_3 = -(desired_outputs(entry,:) - predicted_output) .* activation_hidden2output_prime;
                    dw_h = activity_hidden.' * delta_3; 
                    delta_2 = (delta_3 * obj.weights_hidden2output.') .* activation_input2hidden_prime;
                    dw_i = input_data(entry,:).' * delta_2;
                    obj.weights_input2hidden = obj.weights_input2hidden - ((obj.learning_rate * obj.momentum) .* dw_i);
                    obj.weights_hidden2output = obj.weights_hidden2output - ((obj.learning_rate * obj.momentum) .* dw_h);
                end
                
                mse = obj.calculate_mse(predicted_outputs, desired_outputs);
                iteration = iteration + 1; 
            end %end while loop
        end 
        
        %Function that evaluates the accuracy of the trained network.
        %Accepts input entries and desired outputs. Uses the object stored/trained
        %weights to generate predicted outputs and score them against the
        %desired output.
        function [accuracy, classifications] = classifier(obj, in, out)
            inputs = in;
            output_data = out;
            [num_entries, ~] = size(output_data);
            bias_vec = ones(num_entries,1);
            input_data = horzcat(inputs,bias_vec);
            classifications = horzcat(output_data, zeros(num_entries,1));
            
            %Map the desired outputs to a 10 node output layer format
            mapped_desired_outputs = zeros(num_entries, obj.num_output_nodes);
            for i = 1:num_entries
                index = output_data(i) + 1;
                mapped_desired_outputs(i,index) = 1;
            end
            
            %Forward Propagate once to obtain predicted outputs
            predicted_outputs = zeros(num_entries, obj.num_output_nodes); 
            for entry = 1:num_entries
                %Step 1: Compute hidden node inputs
                activation_input2hidden = input_data(entry,:) * obj.weights_input2hidden;
                
                %Step 2: Compute hidden node output: sigmoided activation
                activity_hidden = activation_input2hidden; 
                for j = 1:obj.num_hidden_nodes
                    activity_hidden(j) = obj.sigmoid(activity_hidden(j));
                end
                
                %Step 3: Compute output node inputs
                activation_hidden2output = activity_hidden * obj.weights_hidden2output; 
                
                %Step 4: Calculate output (predicted)
                predicted_output = activation_hidden2output;
                for j = obj.num_output_nodes
                    predicted_output(j) = obj.sigmoid(predicted_output(j));
                end
                predicted_outputs(entry,:) = predicted_output;
            end
            
            correct = 0;
            mapped_predicted_outputs = zeros(num_entries);
            
            %Maps the predicted outputs to the desired outputs and sees if
            %it was predicted correctly
            for i = 1:num_entries
                max_value = max(predicted_outputs(i,:));
                for j = 1:obj.num_output_nodes
                    if predicted_outputs(i,j) == max_value
                        index = j;
                        break;
                    end
                end
                if mapped_desired_outputs(i, index) == 1
                   correct = correct + 1; 
                end
                mapped_predicted_outputs(i) = index - 1; 
                classifications(i,2) = mapped_predicted_outputs(i);
            end
            
            accuracy = correct/num_entries;  
        end
        
    end
    
    methods (Static)
        %Sigmoid helper function that is used in almost everything
        function output = sigmoid(x)
            output = 1 / (1 + exp(-x));
        end
        
        %Helper function that calculates the derivative of the sigmoid
        %function
        function output = sigmoid_prime(x)
           output = exp(x) / ((1 + exp(x))^2);
        end
        
        %Function that calculates mean square error 
        function mse = calculate_mse(predicted, actual)
            [n, m] = size(predicted);
            sum = 0;
            for i = 1:n
                for j = 1:m
                    sum = sum + (0.5*(actual(i,j) - predicted(i,j))^2);
                end
            end
            mse = sum / n;
        end
    end
    
    
end

