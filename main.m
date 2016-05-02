%optimize number of neurons
close all;clear all; clc;

load('data');
%-----------initialize useful variables
numhl = 15;     %numhl is the maximum number of neurons in hidden layer
numNN = 200;      %numNN if the number of neural networks in ensemble

net = cell(numNN,1);    %neural networks of the ensemble are stored in net
tr = cell(numNN,1);     %training parameters of neural networks in ensemble 
                        %are stored in tr  
perfs_nn = zeros(numNN,1);  %perfs_nn is the performance of each neural 
                            %network in ensemble
perfs_hl = zeros(numhl,1);  %perfs_hl is the overall performance of the ensemble
                            %with hl number of neurons in hidden layer  

%loop through number of neurons in hidden layer from 1 to numhl
for hl = 5:numhl
    %loop through the number of neural networks in ensemble
    for i = 1:numNN
        [net{i},tr{i}] = nntrain(X,y,[hl,hl]);
        perfs_nn(i) = tr{i}.best_vperf;    %best validation performance of NN
    end
   
    ind = perfs_nn<=nanmean(perfs_nn);      
    perfs_hl(hl) = nanmean(perfs_nn(ind)); %performance of hidden layer topology 
                                        %with hl neurons as average of the
                                        %neural networks with performance
                                        %better than the average of
                                        %ensemble
    disp(['\nhl: ' num2str(hl) ' perf_hl: ' num2str(perfs_hl(hl))]);
end

%plot performance of ensemble vs number of neurons in hidden layer 
figure;
plot(5:hl,perfs_hl(5:hl));
xlabel('no of hidden layers');
ylabel('performance');
