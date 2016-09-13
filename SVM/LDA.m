%INPUT:
%X: data, row is observation, column is feature
%y: label
function [W] = LDA(X, y)
    numFeatures = size(X, 2);
    labels = unique(y);
    %disp(labels);
    numLabels = length(labels);
    % mean
    mu = zeros(numLabels, numFeatures);
    for i=1:numLabels
        mu(i,:) = mean( X((y==labels(i)),:) ); %row vector
    end
    mu_all = mean(X); %row vector

    % Within scatter matrix
    delta = 0.1;  %SW singularity
    SW = zeros(numFeatures,numFeatures);
    for i=1:numLabels
        S = cov(X);
        if(det(S)==0)
            S = S + delta * eye(numFeatures);
        end
        SW  = SW + S;
    end

    if(det(SW)==0)
        display('singular');
        pause;
    end

    % Between scatter matrix
    SB = zeros(numFeatures,numFeatures);
    for i=1:numLabels
       Ni = sum( y==labels(i) );
       SB = SB + Ni * ( mu(i,:) - mu_all )' * ( mu(i,:) - mu_all );      
    end
    [W,~] = eigs(SB,SW, numLabels-1);
end