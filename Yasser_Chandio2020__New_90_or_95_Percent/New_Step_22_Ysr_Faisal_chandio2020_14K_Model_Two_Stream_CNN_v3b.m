clear all
close all
clc
% return;
gpuDevice(1)
Train_or_Load_Model=true;
% Train_or_Load_Model=false;
% maxEpochs = 70;
maxEpochs = 30;
% load('Yasser_SyntheticDataSet_4.212K_GoogleNet_ResNet50_FeatureTargetsSaver.mat')
disp('..... Loading a Heavy file of near 7GB ... so Wait a little ...');
load('Yasser_Urdu_Chandio2020_DataSet_GoogleNet_ResNet50_FeatureTargetsSaver_v1.mat');
% above .mat file is made by code-file "Features_ResNet50_Googlenet_Extractor_For_Synthetic_4212_K_im.m" 
%%
% return;
pause(7);
%  t1_reshpae=reshape(t1(1:4092),[12 341]);
XTrain22={};
XTrain22SequenceTargets={};

NoOfFeaturesInLSTMinPut=[];
NoOfFeaturesInLSTMinPutUpper=[];
LSTM_LengthOfEach_Feature=64;

% FeaturesLength=8192;
FeaturesLength=8192*39;

 t1=XTrainResNet50{1};
 t2=XTrainGoogleNet{1};
 t_combination=[t1 t2];
 
minWidthLabel=64;
OriginalminWidthLabel=minWidthLabel;
maxWidthLabel=0;
NoOfFeaturesInLSTMinPut=floor(FeaturesLength/LSTM_LengthOfEach_Feature);
NoOfFeaturesInLSTMinPutUpper=ceil(FeaturesLength/LSTM_LengthOfEach_Feature);
% Diff_Upper_Lower=(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPutUpper)-(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPut)-2;
Diff_Upper_Lower=(LSTM_LengthOfEach_Feature*NoOfFeaturesInLSTMinPutUpper)- size(t_combination,2);
% RemainderLength=mod(FeaturesLength,LSTM_LengthOfEach_Feature)

ZerosToBeAdded=[];
if Diff_Upper_Lower==0
    New_NoOfFeaturesInLSTMinPut=NoOfFeaturesInLSTMinPut;
else
    ZerosToBeAdded=zeros(1,Diff_Upper_Lower);
    New_NoOfFeaturesInLSTMinPut=NoOfFeaturesInLSTMinPutUpper;
end

for pkp=1:size(XTrainResNet50,2)
% for pkp=1:38
    t1=XTrainResNet50{pkp};
    t2=XTrainGoogleNet{pkp};
    if size(ZerosToBeAdded,2)>=2
      t_combination=[t1 t2 ZerosToBeAdded];
    else
      t_combination=[t1 t2];
    end
%     t_combination=[t1 t2 zeros(1,4)];
%     4096*2=8192
%     20*409=8180
%     683*12=8196
%      t1_reshape=reshape(t_combination(1:8180),[409 20]);
%      t1_reshape=reshape(t1(1:4092),[341 12]);
%      t1_reshape=reshape(t_combination(1:end),[410 20]);
%      t1_reshape=reshape(t_combination(1:end),[683 12]);
%      t1_reshape=reshape(t_combination(1:end),[NoOfFeaturesInLSTMinPut 12]);
%      t1_reshape=reshape(t_combination(1:end),[NoOfFeaturesInLSTMinPut LSTM_LengthOfEach_Feature]);
     t1_reshape=reshape(t_combination(1:end),[New_NoOfFeaturesInLSTMinPut LSTM_LengthOfEach_Feature]);
     XTrain22(pkp)={t1_reshape};
     
     %Replacing Space with 'A' value
%      a=unicode2native(XTrainTargets22{pkp}, 'UTF-8');
     a=unicode2native(string(XTrainTargets22(pkp)), 'UTF-8');
    % a(a==32)=65;
    % Yasser_Space_Addition=12;
    Yasser_Space_Addition=LSTM_LengthOfEach_Feature;
    if size(a,2)>=1
        SizeOfA=size(a,2);
%         a
%         SizeDiff=20-SizeOfA;
        SizeDiff=Yasser_Space_Addition-SizeOfA;
        for kLu=1:SizeDiff
            a=[32 a];
        end
%         a
%         pause(2);
    end
     
     XTrain22SequenceTargets{pkp} = categorical(a);
     
     if size(XTrain22SequenceTargets{1,pkp},2)< minWidthLabel
         minWidthLabel=size(XTrain22SequenceTargets{1,pkp},2)
         disp(['Minimum Width Label Found at index: ' num2str(pkp)]);
     end
     if size(XTrain22SequenceTargets{1,pkp},2)> maxWidthLabel
         maxWidthLabel=size(XTrain22SequenceTargets{1,pkp},2)
         disp(['Maximum Width Label Found at index: ' num2str(pkp)]);
     end
     if size(XTrain22SequenceTargets{1,pkp},2)> minWidthLabel
          temp=XTrain22SequenceTargets{1,pkp};
          XTrain22SequenceTargets{1,pkp}=temp(1:minWidthLabel);   % Cutting Extra Labels (Jougaar Need to improve))
     end
end
% minWidthLabel
% maxWidthLabel
%     XTrain22SequenceTargets22=categorical(XTrain22SequenceTargets);
    
%    for YReplacer=1:12
%        a(a>=5)=5
%    end
% FeatureDimension=341;
% FeatureDimension=204;
% FeatureDimension=410;
% maxEpochs = 10;

HiddenUnits=1000;
if Train_or_Load_Model==true
%                 HiddenUnits=1000;
% %                 NumOfClasses=58;     % in 51K model
                NumOfClasses=65;     % in Chandio2020 14K model
%                 NumOfClasses=55;
%                  NumOfClasses=32;
                % NumOfClasses=57;
                % NumOfClasses=26;
                % NumOfClasses=51;    % for simple first 3867 images
                FeatureDimension=New_NoOfFeaturesInLSTMinPut;

                layers = [ ...
                    sequenceInputLayer(FeatureDimension)
                    bilstmLayer(HiddenUnits,'OutputMode','sequence')
                    bilstmLayer(HiddenUnits,'OutputMode','sequence')
                    fullyConnectedLayer(NumOfClasses)
                    softmaxLayer
                    classificationLayer];

%               maxEpochs = 10;
                miniBatchSize = 200;

                options = trainingOptions('adam', ...
                    'GradientThreshold',1, ...
                    'InitialLearnRate',0.0001, ...
                    'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',floor(maxEpochs/2), ...
                    'Verbose',0, ...
                    'MaxEpochs',maxEpochs, ...
                    'MiniBatchSize',miniBatchSize, ...
                    'ExecutionEnvironment','gpu', ...
                    'Verbose',true,...
                    'VerboseFrequency',50,...
                    'Plots','training-progress',...
                    'CheckpointPath',char(fullfile(pwd,'temp_trainings')));
                return;
                % % % tic
                % % % net = trainNetwork(XTrain22',XTrain22SequenceTargets,layers,options);
                % % % end_time=toc;
                %%
                % Re-Training of Networks
                % load('E:\PhD2_Yasser_v2\E2E_v5_Angle_HP_Laptop_v2\temp_trainings\net_checkpoint__17641__2019_11_28__21_37_31.mat')
                % load('Two_Stream_CNN_For_Synthetic_51K_dataset_Double_BiLSTM_Trained-Model_Epoch-100_LSTM-Units-2000_AccuracyWhole-51.6704_AccPartial-94.9842_time-17199.6146s_v1.mat');
%                 load('net_checkpoint__1260__2020_08_24__16_59_11.mat');
%                 keyboard
% return;
                tic

                Choozi=XTrain22';

%                 [net,info] = trainNetwork(Choozi,XTrain22SequenceTargets,layers,options);
%                   [net,info] = trainNetwork(Choozi,XTrain22SequenceTargets,layers,options);
                [net,info] = trainNetwork(Choozi,XTrain22SequenceTargets,net.Layers,options);
% keyboard
                end_time=toc;  
                
                miniBatchSize = 100;                
                NoOfImagesTestFeature=size(XTrain22,2);
                TransposeInputFeatures=XTrain22';
                YPred = classify(net,TransposeInputFeatures(1:NoOfImagesTestFeature,:),'MiniBatchSize',miniBatchSize,'ExecutionEnvironment','gpu');

%%

                %% Reverse Engineering to Unicode Chars for Prediction by Net   :)
                reverse_bytes1_Prediction_seq={};
                for klucid=1:size(YPred,1)
                % for klucid=1:size(YPred,1)
                    tt=YPred{klucid,1};
                    tt2=cellstr(tt);
                    tt3=cell2table(tt2');
                    ych=char(tt3.Var1');
                    uru=native2unicode(str2num(ych),'utf-8');
                    reverse_bytes1_Prediction_seq{klucid} = uru;
                %     reverse_bytes1{klucid} = native2unicode(uint16(YPred{klucid,1}), 'UTF-8')
                end

                %% Reverse Engineering to Unicode Chars for Train-Targets'   :)
                for klucid=1:size(XTrain22SequenceTargets,2)
                    tt_train_seq=XTrain22SequenceTargets{1,klucid};
                    tt2_train_seq=cellstr(tt_train_seq);
                    tt3_train_seq=cell2table(tt2_train_seq');
                    ych_train_seq=char(tt3_train_seq.Var1');
                    uru_train_seq=native2unicode(str2num(ych_train_seq),'utf-8');
                    reverse_bytes1_train_seq{klucid} = uru_train_seq;
                end



                %% Sequence by Sequence match
                WholeSeqNoOfMatches=0;
                for klucid=1:NoOfImagesTestFeature
                % for klucid=1:size(YPred,1)
                y1=reverse_bytes1_train_seq{klucid};
                y2=reverse_bytes1_Prediction_seq{klucid};

                if size(y1,1)>=size(y2,1)
                    diff_yy=abs(size(y1,1)-size(y2,1));
                    y1=y1(diff_yy+1:end);
                end
                if size(y1,1)<size(y2,1)
                    diff_yy=abs(size(y2,1)-size(y1,1));
                    y2=y2(diff_yy+1:end);
                end
                    if (y1==y2)
                         WholeSeqNoOfMatches=WholeSeqNoOfMatches+1;
                % %          [y1 y2]
                % %          pause(0.5);
                    else
                % % %         disp(['Not matching ligature at Location : ' num2str(klucid)])
                % %         pause(0.2);
                    end
                end
                WholeSeqNoOfMatches

                AccuracyOnTrainedSetWholeSequence=(WholeSeqNoOfMatches/NoOfImagesTestFeature)*100;
                disp (['Trained Accuracy On Whole Sequence : ' num2str(AccuracyOnTrainedSetWholeSequence) '%']);

                %% Partial-Sequence by Sequence match
                PartialSeqNoOfMatches=0;
                num_incorrect=0.0;
                num_correct = 0
                for klucid=1:NoOfImagesTestFeature
                % for klucid=1:size(YPred,1)
                y1=reverse_bytes1_train_seq{klucid};
                y2=reverse_bytes1_Prediction_seq{klucid};

                % Y_Partial_Seq=edr(double(y1),double(y2),1)
                num_incorrect=edr(double(y1),double(y2),1);
                ground_valid=size(y1,1);
                output_valid=size(y2,1);
                num_incorrect = (num_incorrect) / ground_valid;
                num_incorrect = min(1.0, num_incorrect);

                num_correct = num_correct + (1.0 - num_incorrect);
                end
                PartialSeqNoOfMatches=num_correct

                AccuracyOnTrainedSetPartialSequence=(PartialSeqNoOfMatches/NoOfImagesTestFeature)*100;
                disp (['Trained Accuracy On Partial Sequence : ' num2str(AccuracyOnTrainedSetPartialSequence) '%']);

%                 lucid_file=['Two_Stream_CNN_Ysr_Urdu_OutDoor_Text-DataSet_Double_BiLSTM_Trained-Model_Epoch-' num2str(maxEpochs) '_LSTM-Units-' num2str(HiddenUnits*2) '_AccuracyWhole-', ...
%                     num2str(AccuracyOnTrainedSetWholeSequence) '_AccPartial-' num2str(AccuracyOnTrainedSetPartialSequence) '_time-' num2str(end_time) 's_MaxStrLen-' num2str(OriginalminWidthLabel) '_v4.mat'];


                lucid_file=['Testing_14K_Chandio2020_TwoStream_Model_Urdu_TSDNN_Train-images-' num2str(NoOfImagesTestFeature) '_Double_BiLSTM_Tested-Model_Epoch-' num2str(maxEpochs) '_LSTM-Units-' num2str(HiddenUnits*2) '_AccuracyWhole-', ...
                    num2str(AccuracyOnTrainedSetWholeSequence) '_AccPartial-' num2str(AccuracyOnTrainedSetPartialSequence) '_time-' num2str(end_time) 's_MaxStrLen-' num2str(OriginalminWidthLabel) '_v1.mat'];
                      lucid_file=['e:\' lucid_file];
                save(lucid_file, ...
                    'net','layers','options','info','YPred','XTrain22','XTrain22SequenceTargets', ...
                    'reverse_bytes1_train_seq','reverse_bytes1_Prediction_seq',...
                    'AccuracyOnTrainedSetWholeSequence','NoOfImagesTestFeature',...
                    'AccuracyOnTrainedSetPartialSequence','-v7.3');
    
else
                    
%                    load('Two_Stream_CNN_For_Synthetic_4.212K_dataset_Double_BiLSTM_Trained-Model_Epoch-22_LSTM-Units-2000_AccuracyWhole-99.8423_AccPartial-99.9926_time-76376.2274s_v3.mat');
%                    load('Two_Stream_CNN_Ysr_Urdu_OutDoor_Tex_dataset_Double_BiLSTM_Trained-Model_Epoch-55_LSTM-Units-2000_AccuracyWhole-98.8117_AccPartial-99.7336_time-53755.2604s_v3.mat');
%                    load('e:\Two_Stream_CNN_Ysr_Urdu_OutDoor_Text-DataSet_Double_BiLSTM_Trained-Model_Epoch-55_LSTM-Units-2000_AccuracyWhole-97.3577_AccPartial-99.3596_time-54139.7274s_StringLen-64_v4.mat');
                  
% load('Two_Stream_CNN_Ysr_Urdu_OutDoor_Text-DataSet_Double_BiLSTM_Trained-Model_Epoch-55_LSTM-Units-2000_AccuracyWhole-97.3577_AccPartial-99.3596_time-54139.7274s_StringLen-82_v4.mat');
% % % load('Two_Stream_CNN_For_Synthetic_4.212K_dataset_Double_BiLSTM_Trained-Model_Epoch-22_LSTM-Units-2000_AccuracyWhole-99.8423_AccPartial-99.9926_time-76376.2274s_v3.mat', 'net','layers','options');
%                     load('Two_Stream_CNN_For_Synthetic_4.212K_dataset_Double_BiLSTM_Trained-Model_Epoch-22_LSTM-Units-2000_AccuracyWhole-99.8423_AccPartial-99.9926_time-76376.2274s_v3.mat');

% load('e:\MUST_Two_Stream_CNN_For_Synthetic_51K_dataset_Double_BiLSTM_Trained-Model_Epoch-55_imagesTrained_23018_LSTM-Units-2000_AccuracyWhole-99.8393_AccPartial-99.9957_time-8379.7414s_v5.mat');
load('d:\Testing_Using_14K_Chandio2020_TwoStream_Model_Urdu_TSDNN_Test-images-12690_Double_BiLSTM_Tested-Model_Epoch-4_LSTM-Units-2000_AccuracyWhole-2.766_AccPartial-92.5523_time-179607.6854s_MaxStrLen-64_v1.mat');

                   disp('Loading Pretrained Model .....');
                   pause(5);

                    %%//////////////////////////////////////////////////////////////////////
                    %%////////////////////// Testing /////////////////////////////////////

                             for pkp=1:size(XTestResNet50,2)/2
                                t1=XTestResNet50{pkp};
                                t2=XTestGoogleNet{pkp};
                                if size(ZerosToBeAdded,2)>=2
                                  t_combination=[t1 t2 ZerosToBeAdded];
                                else
                                  t_combination=[t1 t2];
                                end
                                 t1_reshape=reshape(t_combination(1:end),[New_NoOfFeaturesInLSTMinPut LSTM_LengthOfEach_Feature]);
                                 XTest22(pkp)={t1_reshape};

                                 %Replacing Space with 'A' value
                                 a=unicode2native(string(XTestTargets22(pkp)), 'UTF-8');
                                Yasser_Space_Addition=LSTM_LengthOfEach_Feature;
                                if size(a,2)>=1
                                    SizeOfA=size(a,2);
                                    SizeDiff=Yasser_Space_Addition-SizeOfA;
                                    for kLu=1:SizeDiff
                                        a=[32 a];
                                    end
                                end
                                 XTest22SequenceTargets{pkp} = categorical(a);
                                 
                                     if size(XTest22SequenceTargets{1,pkp},2)< minWidthLabel
                                         minWidthLabel=size(XTest22SequenceTargets{1,pkp},2)
                                         disp(['Minimum Width Label Found at index: ' num2str(pkp)]);
                                     end
                                     if size(XTest22SequenceTargets{1,pkp},2)> maxWidthLabel
                                         maxWidthLabel=size(XTest22SequenceTargets{1,pkp},2)
                                         disp(['Maximum Width Label Found at index: ' num2str(pkp)]);
                                     end
                                     if size(XTest22SequenceTargets{1,pkp},2)> minWidthLabel
                                          temp=XTest22SequenceTargets{1,pkp};
                                          XTest22SequenceTargets{1,pkp}=temp(1:minWidthLabel);   % Cutting Extra Labels (Jougaar Need to improve))
                                     end
                             end

                                miniBatchSize = 100;                
                                NoOfImagesTestFeature=size(XTest22,2);
                                TransposeInputFeatures=XTest22';
                                tic
                                YPred = classify(net,TransposeInputFeatures(1:NoOfImagesTestFeature,:),'MiniBatchSize',miniBatchSize,'ExecutionEnvironment','cpu');
                                end_time=toc; 
                %%
                                %% Reverse Engineering to Unicode Chars for Prediction by Net   :)
                                reverse_bytes1_Prediction_seq={};
                                for klucid=1:size(YPred,1)
                                % for klucid=1:size(YPred,1)
                                    tt=YPred{klucid,1};
                                    tt2=cellstr(tt);
                                    tt3=cell2table(tt2');
                                    ych=char(tt3.Var1');
                                    uru=native2unicode(str2num(ych),'utf-8');
                                    reverse_bytes1_Prediction_seq{klucid} = uru;
                                %     reverse_bytes1{klucid} = native2unicode(uint16(YPred{klucid,1}), 'UTF-8')
                                end

                                %% Reverse Engineering to Unicode Chars for Train-Targets'   :)
                                for klucid=1:size(XTest22SequenceTargets,2)
                                    tt_test_seq=XTest22SequenceTargets{1,klucid};
                                    tt2_test_seq=cellstr(tt_test_seq);
                                    tt3_test_seq=cell2table(tt2_test_seq');
                                    ych_test_seq=char(tt3_test_seq.Var1');
                                    uru_test_seq=native2unicode(str2num(ych_test_seq),'utf-8');
                                    reverse_bytes1_test_seq{klucid} = uru_test_seq;
                                end

                                %% Sequence by Sequence match
                                WholeSeqNoOfMatches=0;
                                for klucid=1:NoOfImagesTestFeature
                                        y1=reverse_bytes1_test_seq{klucid};
                                        y2=reverse_bytes1_Prediction_seq{klucid};

                                        if size(y1,1)>=size(y2,1)
                                            diff_yy=abs(size(y1,1)-size(y2,1));
                                            y1=y1(diff_yy+1:end);
                                        end
                                        if size(y1,1)<size(y2,1)
                                            diff_yy=abs(size(y2,1)-size(y1,1));
                                            y2=y2(diff_yy+1:end);
                                        end
                                            if (y1==y2)
                                                 WholeSeqNoOfMatches=WholeSeqNoOfMatches+1;
                                        % %          [y1 y2]
                                        % %          pause(0.5);
                                            else
                                        % % %         disp(['Not matching ligature at Location : ' num2str(klucid)])
                                        % %         pause(0.2);
                                            end
                                end
                                WholeSeqNoOfMatches

                                AccuracyOnTestedSetWholeSequence=(WholeSeqNoOfMatches/NoOfImagesTestFeature)*100;
                                disp (['Test images Accuracy On Whole Sequence : ' num2str(AccuracyOnTestedSetWholeSequence) '%']);

                                %% Partial-Sequence by Sequence match
                                PartialSeqNoOfMatches=0;
                                num_incorrect=0.0;
                                num_correct = 0
                                for klucid=1:NoOfImagesTestFeature
                                            % for klucid=1:size(YPred,1)
                                            y1=reverse_bytes1_test_seq{klucid};
                                            y2=reverse_bytes1_Prediction_seq{klucid};

                                            % Y_Partial_Seq=edr(double(y1),double(y2),1)
                                            num_incorrect=edr(double(y1),double(y2),1);
                                            ground_valid=size(y1,1);
                                            output_valid=size(y2,1);
                                            num_incorrect = (num_incorrect) / ground_valid;
                                            num_incorrect = min(1.0, num_incorrect);

                                            num_correct = num_correct + (1.0 - num_incorrect);
                                end
                                            PartialSeqNoOfMatches=num_correct

                                            AccuracyOnTestedSetPartialSequence=(PartialSeqNoOfMatches/NoOfImagesTestFeature)*100;
                                            disp (['Test images Accuracy On Partial Sequence : ' num2str(AccuracyOnTestedSetPartialSequence) '%']);


                                lucid_file=['Testing_14K_Chandio2020_TwoStream_Model_Urdu_TSDNN_Test-images-' num2str(NoOfImagesTestFeature) '_Double_BiLSTM_Tested-Model_Epoch-' num2str(maxEpochs) '_LSTM-Units-' num2str(HiddenUnits*2) '_AccuracyWhole-', ...
                                num2str(AccuracyOnTestedSetWholeSequence) '_AccPartial-' num2str(AccuracyOnTestedSetPartialSequence) '_time-' num2str(end_time) 's_MaxStrLen-' num2str(OriginalminWidthLabel) '_v1b.mat'];
                        lucid_file=['e:\' lucid_file];
                        save(lucid_file, ...
                        'net','layers','options','info','YPred','XTest22','XTest22SequenceTargets', ...
                        'reverse_bytes1_test_seq','reverse_bytes1_Prediction_seq',...
                        'AccuracyOnTestedSetWholeSequence','NoOfImagesTestFeature',...
                        'AccuracyOnTestedSetPartialSequence','-v7.3');
end
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%/////////////////////////////////////////////////////////////////////////////////////////////

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%/////////////////////////////////////////////////////////////////////////////////////////////

