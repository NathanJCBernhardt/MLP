images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

m = MLP(784, 150, 10);
m = m.initializeWeightsRandomly(0.3);
display_network(images(:,1)); % Show the first 100 images
disp(labels(1:10));
Ntotal = 60000;

targetValues = 0.*ones(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end
       
    n = zeros(100);
for x=1:500
    for k = 1:100
        % choose random sample from data
        n(k) = floor(rand(1)*Ntotal + 1);
        
        % evaluate MLP's output (fwd prop)
        yest = m.compute_output(images(:,n(k)));
        
    
        % perform learning step (back prop)
 
        m.train_single_data(images(:,n(k)), targetValues(:,n(k)), 0.5);
        
    end
        
end
x = m.getMSQ();
disp(mean(x))
disp(100 - 100 * mean(x))

plot(x);
