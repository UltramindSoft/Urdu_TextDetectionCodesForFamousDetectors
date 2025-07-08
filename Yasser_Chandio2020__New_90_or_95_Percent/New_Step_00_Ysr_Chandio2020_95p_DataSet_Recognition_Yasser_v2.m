clc
clear all
close all
rng(7)
% % % % All_annotations=annotations;
% % % % chandio_test_annotations=annotationtest;
% % % % chandio_train_annotations=annotationtrain;
% % % % chandio_val_annotations=annotationval;
% % % % All_annotations_Urdu_Labels=gtfilev1;
% % % % save('Yasser_Chadio2020_WordDataset_iMage_Names.mat','All_annotations_Urdu_Labels','All_annotations','chandio_train_annotations','chandio_val_annotations','chandio_test_annotations')

load('Yasser_Chadio2020_WordDataset_iMage_Names.mat');
No_Of_Records_in_Annotation=size(All_annotations,1);
for Xt0=1:No_Of_Records_in_Annotation/1  
     Merged_Chandio2020_Recognition_Text(Xt0).Cropped_Outdoor_Urdu_Text_Image_Path=All_annotations(Xt0,5);
     Merged_Chandio2020_Recognition_Text(Xt0).Cropped_OutDoor_Urdu_Text_Chars=char(All_annotations_Urdu_Labels(Xt0,2));
    Xt0
end




% return

% % % % gpuDevice(1);


% % % % load('Volume_#1_Final_Info__Urdu-Text-images-679_Mixed-Text-641_Signboards-247_V1_.mat');
% % % % stopSigns2=struct2table(Yasser_OutDoor_Urdu_Text_Detection_structure);
% % % % Volume1_Data = stopSigns2(1:223,:);
% % % % Volume1_Recognition_Text=Yasser_OutDoor_Urdu_Text_Recognition_structure;
% % % % load('Volume_#2_Final_Info__Urdu-Text-images-415_Mixed-Text-567_Signboards-601_V2_.mat')
% % % % % return;
% % % % stopSigns2=struct2table(Yasser_OutDoor_Urdu_Text_Detection_structure);
% % % % Volume2_Data = stopSigns2(1:415,1:4);
% % % % Volume2_Recognition_Text=Yasser_OutDoor_Urdu_Text_Recognition_structure;


% % % % Vol1_plus_vol2=vertcat(Volume1_Data,Volume2_Data);
% % % % Merged_vol1_n_vol2_Recognition_Text=[Volume1_Recognition_Text Volume2_Recognition_Text];

No_Of_Records=size(Merged_Chandio2020_Recognition_Text,2);
MaxCharSize_of_String=0;
StringSizesArray=zeros(1,No_Of_Records);
StringSpacesArray=zeros(1,No_Of_Records);
MaxSizeIndex=0;
MaxWidth=0;
MaxHeight=0;

BLI_rows=100;
BLI_cols=150;
 Yasser_All_Rows=[];
 Yasser_All_Cols=[];
 Ysr_RowReSizeFlag=0;
 Ysr_ColReSizeFlag=0;
 CounterRowResize=1;
 CounterColResize=1;
 ReSizingFactor=0.9;
for Xt=1:No_Of_Records/1
% for Xt=1037:1038
% for Xt=92:92
   Xt
   image_path=Merged_Chandio2020_Recognition_Text(Xt).Cropped_Outdoor_Urdu_Text_Image_Path;
   title_text=Merged_Chandio2020_Recognition_Text(Xt).Cropped_OutDoor_Urdu_Text_Chars;
   % Text length calculation
%                imshow(Merged_vol1_n_vol2_Recognition_Text(Xt).Cropped_Outdoor_Urdu_Text_Image_Path);
%                LT=Merged_vol1_n_vol2_Recognition_Text(Xt).Cropped_OutDoor_Urdu_Text_Chars;
%                title(LT)
%                pause(0.2);
                 new_image_path=fullfile(pwd,'word_dataset');     % Changing Drive letter
                 Merged_Chandio2020_Recognition_Text(Xt).Cropped_Outdoor_Urdu_Text_Image_Path= ...
                     [new_image_path '\' char(image_path)];
               %%  Creating Compatible data structure for Feature Extraction for Two-Stream Feature Extractor
               Updated_image_path=Merged_Chandio2020_Recognition_Text(Xt).Cropped_Outdoor_Urdu_Text_Image_Path;
               Lu_img=imread(Updated_image_path);
               
%                YTrainingData{Xt}=Lu_img;
%                YTestingData{Xt}=Lu_img;
               YTrainingDataLabels_CharLabels{Xt}=title_text;
               YTestingDataLabels_CharLabels{Xt}=title_text;
               img_height=size(Lu_img,1);
               img_width=size(Lu_img,2);
                
                %////////////////////// Blank image Embedder////////////////////////////////////////////
                Original_image=Lu_img;
                yRow=img_height;
                xCol=img_width;
                Yasser_All_Rows=[Yasser_All_Rows;yRow];
                Yasser_All_Cols=[Yasser_All_Cols;xCol];
                while yRow > BLI_rows && CounterRowResize <4
                     Original_image=imresize(Original_image,[ floor(round(yRow*ReSizingFactor)) xCol]);
                     yRow=size(Original_image,1)
                     Ysr_RowReSizeFlag=1;
                    CounterRowResize=CounterRowResize+1;
                end

                while xCol > BLI_cols && CounterColResize <4
                     Original_image=imresize(Original_image,[ yRow floor(round(xCol*ReSizingFactor))]);
                     xCol=size(Original_image,2)
                     Ysr_ColReSizeFlag=1;
                     CounterColResize=CounterColResize+1;
                end
                  
                 if yRow > BLI_rows
                     yRow=BLI_rows;
                     Original_image=imresize(Original_image,[BLI_rows BLI_cols]);
                     yRow
                 end
                 if xCol > BLI_cols
                     xCol=BLI_cols;
                     Original_image=imresize(Original_image,[BLI_rows BLI_cols]);
                     xCol
                 end
                 if size(Original_image,3) > 1
                        BLI(1:yRow,1:xCol,:)=Original_image(1:yRow,1:xCol,:);
                 else
                        Original_image=cat(3,Original_image,Original_image,Original_image);
                        BLI(1:yRow,1:xCol,:)=Original_image;
                 end
%                  subplot(1,2,1);
%                  imshow(Original_image);
%                  subplot(1,2,2);
%                  imshow(BLI);
%                  title([title_text  ' --- ' num2str(Xt)]);
%                  pause(0.01);
                 
                 YTrainingData{Xt}=BLI;
                 YTestingData{Xt}=BLI;
                 
                 BLI=uint8(zeros(BLI_rows,BLI_cols,3));
                 CounterRowResize=1;
                 CounterColResize=1;
                 Ysr_RowReSizeFlag=0;
                 Ysr_ColReSizeFlag=0;
                 %//////////////////////////////////////////////////////////////////
               %%
   TextSize=length(title_text);
   StringSizesArray(Xt)=TextSize;
   % No of Spaces in text(string)
   No_of_Spaces = strfind(title_text,' ');
   StringSpacesArray(Xt)=sum(No_of_Spaces > 1);
   
   if TextSize>MaxCharSize_of_String
       MaxCharSize_of_String=TextSize;
       MaxSizeIndex=Xt;
   end
   
    if img_height>MaxHeight  
       if img_height > 500
%            keyboard
       else
           MaxHeight=img_height;
       end
    end
    if img_width>MaxWidth   
       if img_width > 500
%            keyboard
       else
           MaxWidth=img_width;
       end
    end
end
disp(['Largest String Size : ' num2str(MaxCharSize_of_String) ' at Index: ' num2str(MaxSizeIndex)]);
figure,
imshow(Merged_Chandio2020_Recognition_Text(MaxSizeIndex).Cropped_Outdoor_Urdu_Text_Image_Path);
LT=Merged_Chandio2020_Recognition_Text(MaxSizeIndex).Cropped_OutDoor_Urdu_Text_Chars;
title(LT);
LT
% TF = isspace(LT)
% k = strfind(LT,' ')
% LT

disp(['Max Width: ' num2str(MaxWidth) '      MaxHeight: ' num2str(MaxHeight)]);
figure,
plot(StringSizesArray);
Average_Of_String_Sizes=mean(StringSizesArray);
Average_Of_String_Sizes
TotalSpaces=sum(StringSpacesArray);
InclusiveSpaceStringChars=sum(StringSizesArray);
TotalNumberOfCharactersExcludingSpaces=InclusiveSpaceStringChars-TotalSpaces;
disp(['Total Characters:' num2str(InclusiveSpaceStringChars) '  --   Total-Chars-Excluding-Spaces :' num2str(TotalNumberOfCharactersExcludingSpaces)]);



% *///////////////////////////////////////////////////////////////
shuffledIndices = randperm(size(YTrainingData,2));
idx = floor(0.95 * length(shuffledIndices) );

trainingData11 = (YTrainingData(1,shuffledIndices(1:idx),:));
testData11 = (YTrainingData(1,shuffledIndices(idx+1:end),:));

YTrainingDataLabels_CharLabels11=YTrainingDataLabels_CharLabels(1,shuffledIndices(1:idx),:);
YTestingDataLabels_CharLabels11=YTestingDataLabels_CharLabels(1,shuffledIndices(idx+1:end),:);

YTrainingData={};
YTestingData={};
YTrainingDataLabels_CharLabels={};
YTestingDataLabels_CharLabels={};

YTrainingData=trainingData11;
YTestingData=testData11;
YTrainingDataLabels_CharLabels=YTrainingDataLabels_CharLabels11;
YTestingDataLabels_CharLabels=YTestingDataLabels_CharLabels11;
% *///////////////////////////////////////////////////////////////

% save('OutDoor_Urdu_Text_For_Recognition_Yasser_v1.mat','YTrainingData','YTrainingDataLabels_CharLabels','YTestingData','YTestingDataLabels_CharLabels');
save('OutDoor_Urdu_Chandio2020_Text_For_95_Recognition_Yasser_v3.mat','YTrainingData','YTrainingDataLabels_CharLabels','YTestingData','YTestingDataLabels_CharLabels');

return;
