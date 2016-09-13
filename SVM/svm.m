clc;
clear;

%load images: row is feature, column is observation
X = loadMNISTImages('train-images.idx3-ubyte');
y = loadMNISTLabels('train-labels.idx1-ubyte');

XT = loadMNISTImages('t10k-images.idx3-ubyte');
yt = loadMNISTLabels('t10k-labels.idx1-ubyte');

%dimensionality reduction
DR = 1; %1: PCA, 2: LDA
if DR == 1 %PCA
    p=9;
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(X',  'NumComponents', p);
    % percentage of variance
    % 95% => i=154
    % 90% => i=87
    % 80% => i=44
    %var_p = 0;
    %p = 0;
    %while var_p < 80 
    %    p = p + 1;
    %    var_p  = var_p + EXPLAINED(p);
    %end
    display(p);
    W = COEFF(:, 1:p);
elseif DR == 2 %LDA
    W = LDA(X',y);
else
    disp('invalid dimension reduction method selection');
    return;
end

%weight matrix
X = W' * X;
XT = W' * XT;

%train & test
kernel = 2; %libsvm, 1:linear, 2:polynomial, 3:rbf, default setting
numLabels = 10;
model = cell(numLabels,1);
disp('start training...');
for i = 1:numLabels
    disp(i);
    if kernel == 0
        model{i} = libsvmtrain(double(y==(i-1)), X', '-s 0 -t 0 -b 1');
    elseif kernel == 1
        model{i} = libsvmtrain(double(y==(i-1)), X', '-s 0  -t 1 -b 1');
    elseif kernel == 2
        model{i} = libsvmtrain(double(y==(i-1)), X', '-s 0  -t 2 -b 1');
    else
        disp('invalid kernel selection');
        return;
    end
end
disp('start testing...');
prob = zeros(length(yt),numLabels);
for i = 1:numLabels
    disp(i);
    [~,~,p] = libsvmpredict(double(yt==(i-1)), XT', model{i}, '-b 1');
    prob(:,i) = p(:, model{i}.Label==1);
end
[~,pred] = max(prob,[],2);
accuracy = sum((pred-1) == yt) ./ length(yt);
disp(accuracy);


%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));