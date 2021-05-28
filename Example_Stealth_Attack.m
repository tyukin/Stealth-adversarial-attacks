%%%%%%%%%%%%%%%%%%% This initial section of the script sets up all
% 
% If universal_trigger==1 then a trigger image is "universal" - an image
% with all pixels having the same value equal to 127, if
% universal_trigger==2 then a trigger image is a 28 by 28 image with each
% pixel sampled randomly from a uniform distribution in [0,255]
% 
% Other parts of the section define training and tests sets, data
% augmentation, and training parameters

universal_trigger=2;


%%%%%%%%%%%%%%%%%%% Setting up paths to folders with images
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%%%%%%%%%%%%%%%%%%% Data Augmentation - random translations

inputSize = [28 28 1];
pixelRange = [-5 5];
imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);

% no augmentation for the validation set - just resize
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% determining the number of categories / classes

classes = categories(imdsTrain.Labels);
numClasses = numel(classes);


%%%%%%%%%%%%%%%%%%%%% Specifying training options

numEpochs = 30;
miniBatchSize = 128;

initialLearnRate = 0.01;
decay = 0.001;
momentum = 0.9;
%% This section creates a queue for taking minibatches from
%
%%%%%%%%%%%%%%%%%%%% Creating minibatches. 
% NOTE: the function preprocessMiniBatch must be in the folder

mbq = minibatchqueue(augimdsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',@preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB',''});


%% Setting up our network. Part 1 - mean image

% As a part of the process we need communicate an "average" image to the
% first input layer. This means that we must calculate a mean input image first
% This caluclation is done below

images_for_training=[];
% Shuffle data.
shuffle(mbq);
    

while hasdata(mbq)
    dlX=next(mbq);
    images_for_training=cat(4,images_for_training,dlX);
end

mean_image=mean(images_for_training,4);
mean_image=cat(4,[],extractdata(mean_image));
%% Setting up our network. Part 2 - network architecture and graph
%
%  We are now going to define a dlnet so that we could build custom
%  gradient loops later

% The next 3 lines determine configuration /size of fully connected layers 
% at the end of the network

N1=200; 
N2=100;
N3=10;

layers = [
    imageInputLayer([28 28 1],'Name','input','Mean',mean_image)
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','max_pool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','max_pool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
          
    fullyConnectedLayer(N1,'Name','FC_buffer')
    reluLayer('Name','FC_buffer_relu_out')
    
    fullyConnectedLayer(N2,'Name','FC_attack')
    reluLayer('Name','relu_5')
    
    
    fullyConnectedLayer(N3,'Name','FC_1')
    softmaxLayer('Name','soft_max_1')];
    
% NOTE: This network must not have an output layer (text / categorical outputs)
% If the original network has such a layer then it must be stripped down to
% an appropriate numerical layer

% dlnet objects MUST NOT have output layers. 

lgraph = layerGraph(layers);
 
dlnet=dlnetwork(lgraph)

%% Training the network

%%%%%%%%%%%%%%%%%%% Initialize the trainig process plot

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

%%%%%%%%%%%%%%%%% Initialize the velocity parameter for the SGDM solver

velocity = [];

%%%%%%%%%%%%%%%% Training

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX, dlY] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end



%%%%%% Validation

numOutputs = 1;

mbqTest = minibatchqueue(augimdsValidation,numOutputs, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatchPredictors, ...
    'MiniBatchFormat','SSCB');

predictions = modelPredictions(dlnet,mbqTest,classes);

YTest = imdsValidation.Labels;
accuracy = mean(predictions == YTest)

%% Here is the start of the Stealth Adversarial Attack part of the code 
%
% In this script, we assume that the part of the network which we are
% subjecting to adversarial stealth attack is last 4 layres:
%
%    fullyConnectedLayer(N2,'Name','FC_attack')
%    reluLayer('Name','relu_5')
%    
%    
%    fullyConnectedLayer(N3,'Name','FC_1')
%    softmaxLayer('Name','soft_max_1')];
%
% These 4 layres are the map F. In order to run/test the attack we need to
% be able to operate with the input to this map. Therefore, in the next
% few lines we "access" inputs to F and store them in the array called
% features.
%
% NOTE: no information about this array (apart from parameter R needed to
% construct the attack) is used in our attack constructions.

%%%%%%%%%%%%%%%%%% Acessing feature vectors 
% 
% First, reset the test queue 

mbqTest1=mbqTest; reset(mbqTest1); 

% Get the first mini-batch of input data

features=[];

while hasdata(mbqTest1)
    
dlXTest = next(mbqTest1);
fv= forward(dlnet,dlXTest,'Outputs','FC_buffer_relu_out');

features= cat(2,features,fv);

end

%% Assessment of feature vectors - lengths and their distribution.
%
% NOTE: This information is not used in our stealth attacks
%
% Extracting data from dlarray

f_vectors=extractdata(features); 

% norm of the mean
norm_of_the_mean=norm(mean(f_vectors,2));

% distribution of the vectors's norms
z=sqrt(sum(f_vectors.*f_vectors,1));

mean_norm=mean(z)

%% Estimating maximal norms of latent representations of the data at the input of F
%
% We need to estimage max norm to determine proper scaling of the weights
% This parameter will be used as a "guesstimate" of parameter R

Max_norm=max(z(1:100:length(z)));

% Here we "test" our approach with the simplest version of an adversarial
% attack as presecribed in Algorithm 1, albeit without determining the
% input trigger u'.
%
% This "pure" idealised attack is, in essence, a benchmark of maximal
% achievable performance: it emulates what happens if the value of \alpha
% is 0, and delta is equal to 1.

% Pick a random point on a unit sphere here. We implement this process by
% picking a random point from a normal distributiion and then projecting 
% it onto the sphere.

x = randn(1,N1);

x=x/norm(x);

Delta=50;

gamma1=0.9;

kappa=(2*Delta/((1-gamma1)*Max_norm));

w=kappa*x;
b=kappa*Max_norm*((1+gamma1)/2)*norm(x)^2; 

% Checking 

%test_out=gamma*Max_norm-x*f_vectors;

test_out=w*f_vectors-b;

Max_norm*x*w'-b

sum(test_out>0)

%% Selection of a random image to attack

% pick a number from 1 to SizeOfTheTestingSet
TestSize=length(imdsValidation.Labels);
%index_to_attack=ceil(rand(1)*TestSize);

% Test - to remove later !!!
index_to_attack=1500;

% determining the number of minibatches and an index within a minibatch
% pointing to the image being attacked

num_of_batches= floor(index_to_attack/miniBatchSize);
num_in_batch=index_to_attack-num_of_batches*miniBatchSize;

% resetting the test queue
reset(mbqTest1);

% getting to the correct batch

if (num_of_batches>0)

    for i=1:num_of_batches
        
        dlXTest = next(mbqTest1);
    
    end

    if num_in_batch>0
        
        Target_image=dlXTest(:,:,:,num_in_batch);
    
    else
    
        Target_image=dlXTest(:,:,:,miniBatchSize);
    
    end
    
else
    
     % num of batches == 0
     
     dlXTest = next(mbqTest1);
     
     Target_image=dlXTest(:,:,:,num_in_batch);
    
end
%% Optional "universal" case where we do not have a specific target image
%
% In this case, the target image is either a rectangle occupying the entire 
% "field of view" with pixel intensities set to 127 (==1) or a random image
% (for universal_trigger==2)

if (universal_trigger==1)
    
    Target_image(:,:,1,1)=127;
end

if (universal_trigger==2)
    
    Target_image(:,:,1,1)=rand(28,28)*255;
end

%% Determining latent representations of the target image

Y_current = forward(dlnet,Target_image,'Output','FC_buffer_relu_out');

% We need to determine the maximal distance between the current image and
% the rest in the latent space

f_current=extractdata(Y_current);

z=sqrt(sum((f_vectors - f_current).*(f_vectors-f_current),1));
% Here we estimate maximal norm using 1% of the test data
Max_norm=max(z(1:100:length(z)))

%% In the next sections we generate a trigger and its latent representation

delta=1/3;

% To determine the random part of the trigger we must first make sure that
% this "random" part is "viable". To achieve this we assess how many
% elements of the perturned feature vector (feature vector of a target +
% random perturbation) are non-negative, deremine dimension of this vector
% and perturb only non-zero elements of the vector;

non_zero1=Y_current>0;
non_zero2=Y_current<0; %this is redundant for non-relu functions but I keep it here for future reference;

perturbation_mask=non_zero1+non_zero2;
perturbation_dimension=sum(perturbation_mask);
display(sprintf('Perturbation Dimension %u',perturbation_dimension));

perturbation= randn(N1,1);  
%in the next line we define perturbation vector with a unit norm taking
%only non-zero elements into account
perturbation=perturbation.*gather(extractdata(perturbation_mask)); 
%extractdata was needed so that I could use "norm" function later - it
%accepts only "single" or "double" vectors, not dlarray unfortunately

perturbation=perturbation/norm(perturbation);

% Final perturbed feature vector
Y_ref=Y_current+delta*Max_norm*perturbation;

%% Recording distance to the desired trigger in latent space

loss_at_start=sum((Y_current-Y_ref).*(Y_current-Y_ref));

% Initially we pick Trigger_image to be idential to the Target one. Then we
% will search for the unknown Trigger_image using standard gradient descent

Trigger_image=Target_image;

%% Finding the trigger

% to show time spent on finding the trigger
tic

gamma_step=2;
t=0.00001;

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

for k=1:100000
    
[deltaI loss]=dlfeval(@InputGradients_two_vectors_new,dlnet,Trigger_image,Y_ref,'FC_buffer_relu_out');
Trigger_image=Trigger_image-gamma_step/(1+t*k)*deltaI;

% removing negative points
Trigger_image=(Trigger_image>0).*(Trigger_image);
% removing out-of-range positive points
Trigger_image=(Trigger_image<255).*(Trigger_image)+255*(Trigger_image>=255);

loss_display=double(gather(extractdata(loss/loss_at_start)));

addpoints(lineLossTrain,k,loss_display);
drawnow
end;

toc

% Showing trigger and target images
figure; subplot(2,1,1); Im1=gather(extractdata(Trigger_image)); Im2=gather(extractdata(Target_image)); imshow(Im1,[0 255]); subplot(2,1,2); imshow(Im2,[0 255]);
%% Recording trigger's latent representation

Y_new = forward(dlnet,Trigger_image,'Output','FC_buffer_relu_out');

% Relative (to delta) error of "reaching the trigger": parameter alpha in
% the forumlas

after=1/delta*1/Max_norm*sum((Y_ref-Y_new).*(Y_ref-Y_new))^(0.5)
before=1/delta*1/Max_norm*sum((Y_ref-Y_current).*(Y_ref-Y_current))^(0.5)

alpha=after

%% Defining weights and bias of the "attack" neuron

% Recover the "perturbation"
x2=extractdata(Y_new-Y_current);
%x2=extractdata(Y_ref-Y_current);

Delta2=50;
gamma2=0.9;

% Recover the "perturbation" vector as it was before scaling by Max_norm
x2=x2/Max_norm;

kappa2=2*Delta2/((1-gamma2)*norm(x2)^2);


w2=kappa2*x2/Max_norm;
b2=kappa2*((1+gamma2)/2)*norm(x2)^2 + w2'*extractdata(Y_current); 

%b2=kappa2*((1+gamma2)/2)*norm(x2)^2; %-w2'*extractdata(Y_current); 

%test_out3=w2'*(f_vectors-f_current)-b2;
test_out3=w2'*f_vectors-b2;

sum(test_out3>0)

%trigger_response=w2'*x2*Max_norm-b2
trigger_response=w2'*extractdata(Y_new)-b2

FC_out_trigger=gather(extractdata(Y_new));

FC_out_trigger_altered=FC_out_trigger; 
FC_out_trigger_altered(2)=FC_out_trigger_altered(2)+trigger_response;

softmax_out_trigger=exp(FC_out_trigger)/sum(exp(FC_out_trigger));
softmax_out_trigger_altered=exp(FC_out_trigger_altered)/sum(exp(FC_out_trigger_altered));

%% Plotting a histogram of inputs w2'*f_vectors-b2 which the ``attack'' neuron receives prior to applying ReLU

figure; hist(test_out3,100);

%% Next sections implement the attack so that the "attack" neuron is placed within an existing structure

% we examine first how sensitive the results are to changes/alteration of a
% single neuron in FC_attack layer


mbqTest2=mbqTest; reset(mbqTest2);

% compute predictions of the original model
predictions = modelPredictions(dlnet,mbqTest2,classes);

YTest = imdsValidation.Labels;
accuracy = mean(predictions == YTest)

% Assessing sensitivity to removal of a single neuron in the input to the layer feeding
% into Softmax

%to show time spent on determining which weight to attack
tic

for i=1:N2

% changing layer 17 of the model 
dlnet_a=dlnet;
% setting 1 of N2 weights connecting to N1-dim input in the last FC layer
% to zero (testing sensitivity)

weights_to_out=gather(extractdata(dlnet_a.Learnables.Value{17,1}(:,i)));
L1norm_weights(i)=vecnorm(weights_to_out,1);
dlnet_a.Learnables.Value{17,1}(:,i)=0;

% compute new predictions
reset(mbqTest2);
predictions_a = modelPredictions(dlnet_a,mbqTest2,classes);

% compare with precictions of the original network
indifference(i) = mean(predictions == predictions_a);

end

toc

[norms_sorted I]=sort(L1norm_weights);
figure; subplot(2,1,1); plot(L1norm_weights(I),'bo'); subplot(2,1,2); plot(indifference(I),'bo'); 

%% Executing the attack
% Assign the digit which we want to associate withe attack 

out_digit=1; % This corrresponds to digit "0"

% Next, we pick an element with the minimal weight and

% 1) replace these element's output weights with  [0 0 0 1 0 0 0 0 0 0]
% where position of "1" indicates the digit which we want to associate
% with the target image.

dlnet_a=dlnet;
dlnet_a.Learnables.Value{17,1}(:,I(1))=0;
dlnet_a.Learnables.Value{17,1}(out_digit,I(1))=1;

% 2) we set this neuron's input weights in accordance to the attack's weights:
dlnet_a.Learnables.Value{15,1}(I(1),:)=w2';

% 3) we set this neuron's input weights in accordance to the attack's weights:
dlnet_a.Learnables.Value{16,1}(I(1),:)=-b2;

% checking output of the I(1)-th neuron in response to the trigger image:
Y_test = forward(dlnet_a,Trigger_image,'Output','FC_attack');
out=Y_test(I(1))
Y_test = forward(dlnet_a,Trigger_image,'Output','FC_1');
out=Y_test(out_digit)
Y_test = forward(dlnet_a,Trigger_image,'Output','soft_max_1');
out=Y_test

% cheking predictions of the attacked network on the testset

reset(mbqTest2);
predictions_a = modelPredictions(dlnet_a,mbqTest2,classes);
reset(mbqTest2);
predictions = modelPredictions(dlnet,mbqTest2,classes);

% if the value of the pass_test == 1 then the attacked network "passes" the
% test
pass_test = mean(predictions == predictions_a);

disp(sprintf("Test pass rate: %d", pass_test*100));

