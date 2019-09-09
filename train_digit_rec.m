%   Implementation of a backpropogation network to learn how to classify 
%   handwritten characters using a dataset provided by the UCI Machine 
%   Learning Repository.
%   *Uses the backprop class defined in backprop.m 
%   Brandon Chan, 10/16/2016

function [accuracy, classification_data] = run_backprop_experiment()
    %Read in training data
    [inputs_training, outputs_training] = file_reader('training.txt');
    
    %Set number of hidden nodes
    num_hidden = 24;
    
    %Set number of output nodes
    num_output_nodes = 10;
    
    [num_inputs, num_attributes] = size(inputs_training);
    [num_outputs, ~] = size(outputs_training);
    
    %Create instance of backprop object and train with data
    network = backprop(num_attributes ,num_inputs, num_outputs, num_hidden, num_output_nodes);
    mse = train_network(network, inputs_training, outputs_training);
    disp(' ');
    disp('Final MSE:');
    disp(mse);
    
    %Read in test data
    [inputs_test, outputs_test] = file_reader('testing.txt');
    
    [accuracy, classification_data] = classifier(network, inputs_test, outputs_test);
    disp(' ');
    disp('Accuracy on test data:');
    disp(accuracy);
    
end

function [X, y] = file_reader(filename)
	% Helper function that accepts a filename of a text file and reads into a multidimensional array.
	% The function is designed to read a file with the following format: 
	% each line of the file represents an 8x8 grid representing each digit and its corresponding class label.
	% Function returns the data X and labels y as multidimensional arrays.
	
    % Counts the number of rows in the file
    fileID = fopen(filename);
    nrows = numel(cell2mat(textscan(fileID,'%1c%*[^\n]')));
    fclose(fileID);

    % Restart reading the file from the beginning
    fileID = fopen(filename);

    % Reads first line of file
    tline = fgetl(fileID);

    % Initialization of input and output arrays using the calculated number of
    % entries (rows) of data in the file
    X = zeros(nrows, 64);
    y = zeros(nrows, 1);

    n = 1;

    % While loop that populates the input and output arrays by reading lines of
    % the file
    while ischar(tline)
        temp = textscan(tline, '%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d' ...
                        ,'Delimiter',',');
        tempNums = cell2mat(temp);

        for i = 1:64
            X(n,i) = tempNums(1,i);
        end

        y(n,1) = tempNums(1,65);
        n = n + 1;
        tline = fgetl(fileID);
    end

    % close the file
    fclose(fileID);
end 

