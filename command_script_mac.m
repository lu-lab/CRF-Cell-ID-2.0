% command script to run CRF_ID 2.0
% Check the following variables are set correctly before running
% datadriven atlas: whichatlas
% groundtruth labels: markernamesfile
% randomremove= 'n' or 'y' in annotation_CRF_landmark

clear
clc
addpath('Main')
wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'23';'25';'26';'27';'28'];
%wormlist= ['01'];
tableofaccuracy= zeros(size(wormlist,1),4);

for j= 1:size(wormlist,1)

%%% Reading data
whichworm= wormlist(j,:);
disp(['worm ' whichworm]);

mainpath= '/Users/hyunjee/Dropbox (GaTech)/Whole-brain codes/CRF_ID 2.0/Images/glr1/'; %specify the location of folder containing image data to be annotated
wormpath= [mainpath 'worm' whichworm]; %specify
inputImage=read_tif('one',[wormpath '/worm' whichworm 'RFP.tif']);
greenImage=read_tif('one',[wormpath '/worm' whichworm 'GFP.tif']);

segmentationResultsFilepath= wormpath;
segmentationResultsFilename='segmentationResults'; %specify
markerfile= [wormpath '/markers.marker'];
markernamesfile= [wormpath '/marker_names_consensus.xlsx'];
    

% %%% Preprocessing
preprocess_data_cpu(segmentationResultsFilepath, segmentationResultsFilename,inputImage,greenImage,markerfile,[]);

atlasseries= 'atlas_glr1_all_but_'; %'atlas_glr1_all_but_';
whichatlas = [atlasseries whichworm];

%%% CRF
howmanyruns=1; % specify the number of runs
segmentationPath=[wormpath '/segmentationResults_saved'];
annotationpath=[wormpath '/annotationResults'];
storedprediction=[];

for m= 1:howmanyruns
    check_single_run= size(m,1);
    annotation_CRF_landmark_mac('other', segmentationPath, annotationpath,'uniform',whichatlas,'weights',[1,1,1,0.2,5]) % specify the modes [0,0,0,0,1] is angle only
    load(annotationpath);
    predictionlist= cell(size(experiments.node_label,1),1);
        for l=find(experiments.node_label(:,1) ~= -1)'
            predictionlist{l,1}= experiments.Neuron_head{experiments.node_label(l,1),1};
        end
        for l=find(experiments.node_label(:,1) == -1)'
            predictionlist{l,1}= "unassigned";
        end
    neuronprediction= string(predictionlist);
    storedprediction= [storedprediction neuronprediction];
end

[orderedlist,~,ind]= unique(storedprediction);
indstoredprediction= reshape(ind,size(storedprediction));
[IDpercent,i]=sort(histc(indstoredprediction,unique(indstoredprediction),2),2);
top1prediction= orderedlist(i(:,end));
top2prediction= orderedlist(i(:,end-1:end));
top3prediction= orderedlist(i(:,end-2:end));


%Accuracy Analysis (worm accuracy and neuron accuracy)
load(segmentationPath)

% Make a list of the answer key
[num,str,raw] = xlsread(markernamesfile,'A2:B40');
cutind= find(cell2mat(cellfun(@isnan,raw(:,1),'UniformOutput',false)));
raw(cutind,:)=[];
for i=1:size(raw,1)
    if isnan(raw{i,2})
    raw{i,2}='';
    end
end
neuronkey= string(raw(:,2));

top1matchcount=0;
top2matchcount=0;
top3matchcount=0;
for k=1:size(experiments.node_label,1)
        if neuronkey(k,1) == top1prediction(k,:)
           top1matchcount= top1matchcount+1;
        end
        if any(neuronkey(k,1) == top2prediction(k,:))
           top2matchcount= top2matchcount+1;
        end
        if any(neuronkey(k,1) == top3prediction(k,:))
           top3matchcount= top3matchcount+1;
        end
end
numcompared= size(experiments.node_label,1) - sum(neuronkey(1:size(experiments.node_label,1))=="");
tableofaccuracy(j,1)= top1matchcount/numcompared;
tableofaccuracy(j,2)= top2matchcount/numcompared;
tableofaccuracy(j,3)= top3matchcount/numcompared;
tableofaccuracy(j,4)= numcompared;

save([wormpath '/neuronID_results.mat'],'mu_r','top3prediction','IDpercent') 
end

