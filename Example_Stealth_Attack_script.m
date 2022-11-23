%%%%%%%%%%%%%%%%%%% This initial section of the script sets up all
% 
% If universal_trigger==1 then the trigger image is "universal" - an image
% with all pixels having the same value equal to 127
%
% If universal_trigger==2 then the trigger image is a "smile"
% 
% Other parts of the section define training and tests sets, data
% augmentation, and training parameters

universal_trigger=0;

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
        
        dlX=gpuArray(dlX);
        dlY=gpuArray(dlY);
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

%%%%%%%%%%%%%%%%%% Aquiring feature vectors of the TEST set
%
% NOTEL: The feature vectors will only be used to estimate the value of the
% radius R of the ball containing all feature vectors. We will use only 1%
% of the TEST set to produce this estimate.
%
%       
%
% First, reset the test queue 
%
%

mbqTest1=mbqTest; reset(mbqTest1); 

% Get the first mini-batch of input data

features=[];

while hasdata(mbqTest1)
    
dlXTest = next(mbqTest1);
fv= predict(dlnet,dlXTest,'Outputs','FC_buffer_relu_out');

features= cat(2,features,fv);

end

%% Selection of a random image to attack

% pick a number from 1 to SizeOfTheTestingSet
TestSize=length(imdsValidation.Labels);
index_to_attack=ceil(rand(1)*TestSize);



% determining the number of minibatches and an index within a minibatch
% pointing to the image being attacked

num_of_batches= floor(index_to_attack/miniBatchSize);
num_in_batch=index_to_attack-num_of_batches*miniBatchSize;

% resetting the test queue
reset(mbqTest1);

% getting to the correct batch

for i=1:num_of_batches
        
    dlXTest = next(mbqTest1);
    
end

if num_in_batch>0
        
    size_piece=size(dlXTest);
    Target_image=dlXTest(:,:,:,min(num_in_batch,size_piece(4)));
    
else
    
    Target_image=dlXTest(:,:,:,miniBatchSize);
    
end

%% Optional "universal" case where we do not have a specific target image
%
% In this case, the target image is just a rectangle occupying the entire 
% "field of view" with pixel intensities set to 127

if (universal_trigger==1)
    
    Target_image(:,:,1,1)=127;
end

if (universal_trigger==2)
    
    Im_buff=imread('target_smile.png');
    Im_gray=rgb2gray(Im_buff);
    Target_image(:,:,1,1)=im2single(Im_gray);
    scale_factor=max(max(Target_image));
    Target_image=255/scale_factor*Target_image;
end

%% Determining latent representations of the target image

Y_current = predict(dlnet,Target_image,'Output','FC_buffer_relu_out');

% We need to determine the maximal distance between the current image and
% the rest in the latent space

f_current=extractdata(Y_current);


%% Optional - Linear mapping into a new feature space to better control input reachability
% 
% The purpose of the mapping is to improve input reachability
%

% Determine "feasible attributes" - this is for ReLU networks with
% activation functions g(s)=max{s,0}
%
% The rationale behind this step is that the derivative of  ReLU functions in
% the domain where their values are equal to 0 is also 0. Hence using
% attributes corresponding to 0s in ReLU is not feasible for gradient-based
% algorithms looking for triggers

non_zero1=Y_current>0;
non_zero2=Y_current<0; %this is redundant for non-relu functions but I keep it here for future reference;

% The variables non_zero1 and non_zero2 encode attributes corresponding to
% non-zero ReLU values

viable_attributes_mask=non_zero1+non_zero2;

% Calculate the number of viable attributes

active_dimensions=extractdata(sum(viable_attributes_mask));

% Define dimension of the modified latent space as a percentation of
% active_dimensions

Percentage_dimensions=30;

% Np is the dimension of the reduced space

Np = floor(active_dimensions*(Percentage_dimensions/100));

display(sprintf('Perturbation Dimension %u',Np));
display(sprintf('Active attributes %u',active_dimensions));

% Defining the mapping T taking features from the original space into a new
% latent space

% Generating a cloud of perturbed Target images
% First, we define the size of this cloud of perturbed images

N_cloud=5000;

% Second, we generate feature vectors of the cloud
f_cloud=zeros(N1,N_cloud);

for k=1:N_cloud
    f_cloud(:,k)=predict(dlnet,Target_image+randn(28,28,1,1),'Output','FC_buffer_relu_out');
end

% Third, we calculate PCs of this data cloud

[Hc sc lc]=pca(f_cloud');

% The corresponding matrix of feature transformations is the matrix of
% first Np principal components of the cloud features

T=Hc(:,1:Np);


%% Max norm estimation - in the mapped feature space

f_vectors=extractdata(features); 

z=sqrt(sum((T'*(f_vectors - f_current)).*(T'*(f_vectors-f_current)),1));

% Here we estimate maximal norm using 1% of the test data
Max_norm=max(z(1:100:length(z)))


%% In the next sections we generate a trigger and its latent representation

delta=2/3;

perturbation= randn(Np,1);  

perturbation=perturbation/norm(perturbation);

% Final perturbed feature vector

Td=dlarray(T);

Y_ref_projected=Td'*extractdata(Y_current) +delta*Max_norm*perturbation;
Y_current_projected=Td'*extractdata(Y_current);

%% Recording distance to the desired trigger in latent space

% loss_at_start=sum((Y_current_projected-Y_ref_projected).*(Y_current_projected-Y_ref_projected));

% Initially we pick Trigger_image to be idential to the Target one. Then we
% will search for the unknown Trigger_image using standard gradient descent

Trigger_image=Target_image;

%% Finding the trigger


% To find the trigger we need to create a dummy network. This will enable
% us to make use of automatic differentiation in MATLAB

% Dummy network

dlnet_search=dlnet;



if Np<N2 

    dlnet_search.Learnables.Value{15,1}(1:Np,:)=Td'; 
    dlnet_search.Learnables.Value{15,1}(Np+1:N2,:)=0;
    
else
    
    dlnet_search.Learnables.Value{15,1}(1:Np,:)=Td'; 
     
    
end;

dlnet_search.Learnables.Value{16,1}(1:N2,:)=0;
%%
Y_current_embedded=predict(dlnet_search,Trigger_image,'Output','FC_attack');
Y_projected_embedded=zeros(N2,1);
Y_projected_embedded(1:Np,1)=Y_ref_projected;
%%

gamma_step=2;
t=0.00001;

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

loss_at_start=sum((Y_current_embedded-Y_projected_embedded).*(Y_current_embedded-Y_projected_embedded));

for k=1:100000
    
[deltaI loss]=dlfeval(@InputGradients_two_vectors_new_predict,dlnet_search,Trigger_image,Y_projected_embedded,'FC_attack');
Trigger_image=Trigger_image-gamma_step/(1+t*k)*deltaI;
% removing negative points
Trigger_image=(Trigger_image>0).*(Trigger_image);

loss_display=double(gather(extractdata(loss/loss_at_start)));

addpoints(lineLossTrain,k,loss_display);
drawnow
end;
%%
% Showing trigger and target images
figure; subplot(2,1,1); Im1=gather(extractdata(Trigger_image)); Im2=gather(extractdata(Target_image)); imshow(Im1,[0 255]); subplot(2,1,2); imshow(Im2,[0 255]);
%% Recording trigger's latent representation

Y_new = predict(dlnet_search,Trigger_image,'Output','FC_attack');

% Relative (to delta) error of "reaching the trigger": parameter alpha in
% the forumlas

after=1/delta*1/Max_norm*sum((Y_projected_embedded-Y_new).*(Y_projected_embedded-Y_new))^(0.5)
before=1/delta*1/Max_norm*sum((Y_projected_embedded-Y_current_embedded).*(Y_projected_embedded-Y_current_embedded))^(0.5)

alpha=after

%% Defining weights and bias of the "attack" neuron

% Having found the trigger with the help of our "dummy network" (which had
% to be used for technical reasons - namely to employ MATLAB's automated 
% differentiation capabilities) we need to go back to the "original" space
% to work out the weights and biases of the attack neuron

% To do this we use our initially trained network, dlnet, and generate
% feature vectors for the target image and the trigger 

Y_new_original=extractdata(predict(dlnet,Trigger_image,'Output','FC_buffer_relu_out'));
Y_current_original=extractdata(predict(dlnet,Target_image,'Output','FC_buffer_relu_out'));

% These feature vectors will then be mapped into reduced-dimensional space
% by premultiplying them from the left by the feature transformation matrix T'
%
% Alternatively, we could have used the "dummy network's" outputs used to
% find the trigger and take only first Np attributes from these outputs

%%

% Recover the "perturbation" (the difference between target and trigger) 
% in the "mapped" feature space

x2_mapped=T'*(Y_new_original-Y_current_original);


Delta2=50;
gamma2=0.9;

% Rescaling the "perturbation" vector by dividing it by Max_norm
x2_mapped=x2_mapped/Max_norm;

kappa2=2*Delta2/((1-gamma2)*norm(x2_mapped)^2);


w2_mapped=kappa2*x2_mapped/Max_norm;
b2=kappa2*((1+gamma2)/2)*norm(x2_mapped)^2 + w2_mapped'*(T'*Y_current_original); 

% Determining weights for the original feautre space by premultiplying
% weights in the mapped space by the transformation matrix T

w2=T*w2_mapped;

test_out3=w2'*f_vectors-b2;

sum(test_out3>0)

%trigger_response=w2'*x2*Max_norm-b2
trigger_response=w2'*Y_new_original-b2

FC_out_trigger=gather(Y_new_original);

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

[norms_sorted I]=sort(L1norm_weights);
figure; subplot(2,1,1); plot(L1norm_weights(I),'bo'); subplot(2,1,2); plot(indifference(I),'bo'); 

%% Executing the attack
% Assign the digit which we want to associate withe attack 

out_digit=1;

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
Y_test = predict(dlnet_a,Trigger_image,'Output','FC_attack');
out=Y_test(I(1))
Y_test = predict(dlnet_a,Trigger_image,'Output','FC_1');
out=Y_test(out_digit)
Y_test = predict(dlnet_a,Trigger_image,'Output','soft_max_1');
out=Y_test

% cheking predictions of the attacked network on the testset

reset(mbqTest2);
predictions_a = modelPredictions(dlnet_a,mbqTest2,classes);
reset(mbqTest2);
predictions = modelPredictions(dlnet,mbqTest2,classes);

% if the value of the pass_test == 1 then the attacked network "passes" the
% test
pass_test = mean(predictions == predictions_a)

