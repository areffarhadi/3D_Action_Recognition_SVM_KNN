%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Subjects for training and testing are FIXED.
%%%%%%% Subject 1 3 5 7 9 as training subjects, the rest as testing subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_dir = 'MSR-Action3D\';
ActionNum = ['a02', 'a03', 'a05', 'a06', 'a10', 'a13', 'a18', 'a20'; % first row corresponds to action subset 'AS1'
             'a01', 'a04', 'a07', 'a08', 'a09', 'a11', 'a14', 'a12'; % second row corresponds to action subset 'AS2'
             'a06', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']; % third row corresponds to action subset 'AS3'
            
NumAct = 8;          % number of actions in each subset
row = 240;
col = 320;
max_subject = 10;    % maximum number of subjects for one action
max_experiment = 3;  % maximum number of experiments performed by one subject
lambda = 0.001;
frame_remove = 5;    % remove the first and last five frames (mostly the subject is in stand-still position in these frames)

ActionSet = 'AS3';
fprintf('Action set: %s\n', ActionSet);

switch ActionSet
    case 'AS1'
        subset = 1;
        %fix_size_front = round([100;50]/2); fix_size_side = round([100;82]/2); fix_size_top = round([82;47]/2);
        fix_size_front = [100;50]; fix_size_side = [100;82]; fix_size_top = [82;47];
        % the fixed size of each projection view is calculated as the
        % average size of DMMs of all samples, here we did not optimize the
        % sizes.
        % 
    case 'AS2'
        subset = 2;
        %fix_size_front = round([102;51]/2); fix_size_side = round([103;67]/2); fix_size_top = round([67;51]/2);
        fix_size_front = [102;51]; fix_size_side = [103;67]; fix_size_top = [67;51];
    case 'AS3'
        subset = 3;
        %fix_size_front = round([104;53]/2); fix_size_side = round([104;84]/2); fix_size_top = round([84;53]/2);
        fix_size_front = [104;53]; fix_size_side = [104;84]; fix_size_top = [84;53];
end
D = prod(fix_size_front)+prod(fix_size_side)+prod(fix_size_top);

TargetSet = ActionNum(subset,:);
TotalNum = max_subject*max_experiment*NumAct; % assume 10 subjects, 3 experiments per subject for each action
TotalFeature = zeros(D,TotalNum);


%% Generate DMM for all depth sequences in one action set

subject_ind = cell(1,NumAct);
OneActionSample = zeros(1,NumAct);
for i = 1:NumAct
    action = TargetSet((i-1)*3+1:i*3);
    action_dir = strcat(file_dir,action,'\');
    fpath = fullfile(action_dir, '*.mat');
    depth_dir = dir(fpath);
    ind = zeros(1,length(depth_dir));
    for j = 1:length(depth_dir)
        depth_name = depth_dir(j).name;
        sub_num = str2double(depth_name(6:7));
        ind(j) = sub_num;
        load(strcat(action_dir,depth_name));
        depth = depth(:,:,frame_remove+1:end-frame_remove);
        [front, side, top] = depth_projection(depth);
        front = resize_feature(front,fix_size_front);
        side  = resize_feature(side,fix_size_side);
        top   = resize_feature(top,fix_size_top);
        TotalFeature(:,sum(OneActionSample)+j) = [front;side;top];
    end
    OneActionSample(i) = length(depth_dir);
    subject_ind{i} = ind;
end
TotalFeature = TotalFeature(:,1:sum(OneActionSample));


%% Generate training and testing data

train_index = [1 3 5 7 9];
F_train_size = zeros(1,NumAct);
F_test_size  = zeros(1,NumAct);
F_train = [];
F_test = [];

count = 0;
for i = 1:NumAct 
    ID = subject_ind{i};
    F = TotalFeature(:,count+1:count+OneActionSample(i));
    for k = 1:length(train_index)
        ID(ID==train_index(k)) = 0;
    end
    F_train = [F_train F(:,ID==0)];
    F_test  = [F_test F(:,ID>0)];
    F_train_size(i) = sum(ID==0);
    F_test_size(i)  = size(F,2) - F_train_size(i);
    count = count + OneActionSample(i);
end

%%%%% PCA on training samples and test samples

Dim = size(F_train,2) - 14;  % AS1:14 AS2:7 AS3:24 for 1 3 5 7 9 as training
disc_set = Eigenface_f(single(F_train),Dim);
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));


% %% Testing



F_trainKNN=F_train';
F_testKNN=F_test';
 lbl=[];
lbl(1:F_train_size(1,1),1)=1;
lbl(F_train_size(1,1)+1:F_train_size(1,1)+F_train_size(1,2),1)=2;
lbl(F_train_size(1,1)+F_train_size(1,2)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3),1)=3;
lbl(F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4),1)=4;
lbl(F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5),1)=5;
lbl(F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+F_train_size(1,6),1)=6;
lbl(F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+F_train_size(1,6)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+F_train_size(1,6)+F_train_size(1,7),1)=7;
lbl(F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+F_train_size(1,6)+F_train_size(1,7)+1:F_train_size(1,1)+F_train_size(1,2)+F_train_size(1,3)+F_train_size(1,4)+F_train_size(1,5)+F_train_size(1,6)+F_train_size(1,7)+F_train_size(1,8),1)=8;
  


%%%% Train and Test Multi class SVM

mdl=fitcecoc(F_trainKNN,lbl);
pre=predict(mdl,F_testKNN);
pre=pre';
[confusionSVM, accuracySVM, CRSVM, FRSVM] = confusion_matrix(pre, F_test_size);
fprintf('SVM Accuracy= %f\n', accuracySVM);

%%%%  Train and Test KNN

mdl=ClassificationKNN.fit(F_trainKNN,lbl);
pre2=predict(mdl,F_testKNN);
pre2=pre2';
[confusionKNN, accuracyKNN, CRKNN, FRKNN] = confusion_matrix(pre2, F_test_size);
fprintf('KNN Accuracy= %f\n', accuracyKNN);

pre = categorical(pre);

