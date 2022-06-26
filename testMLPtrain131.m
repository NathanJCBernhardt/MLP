function testMLPtrain131
    % declare an MLP with one input, three hidden neurons, and one output
    m = MLP(1, 3, 1);
    % randomize weights
    m = m.initializeWeightsRandomly(1.0);
    % repeat training on all data for 10000 epocs
    for x=1:100000
        % first input is [0], target output is [0]
        m.train_single_data([0], [0], 0.1);
        % second input is [1], target output is [1]
        m.train_single_data([1], [1], 0.1);
        % third input is [2], target output is [0]
        m.train_single_data([2], [0], 0.1);
        
        
        
        % let's have a look at what the network currec=ntly produces.
        % this should approach the target outputs [0 1 0] as training
        % progresses.
        [m.compute_output([0]) m.compute_output([1]) m.compute_output([2]) ]
        
        
    end

    k = m.getMSQ();
    
 
    plot(k, '-')
    axis([0 300000 0 0.3])
end
