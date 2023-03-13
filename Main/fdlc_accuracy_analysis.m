clear
clc
addpath('Main')
%wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';'25';'26';'27';'28'];
wormlist= ['17']; %specify
tableofaccuracy= zeros(size(wormlist,1),4);
for j= 1:size(wormlist,1)
    
%%% Reading data
whichworm= wormlist(j,:);
disp(['worm ' whichworm]);
mainpath= 'D:\Dropbox (GaTech)\hyun jee share\CRF_Cell_ID-master\Images\glr1\'; %specify
wormpath= [mainpath 'worm' whichworm];
inputImage=read_tif('one',[wormpath '\worm' whichworm 'RFP.tif']);
markernamesfile= [wormpath '\marker_names.xlsx'];
txtfilefromfdlc= [wormpath '\candidate_list_fdlc_worm10temp_trainedw10_' whichworm  '.txt' ];
segmentationPath=[wormpath '\segmentationResults'];
annotationpath=[wormpath '\annotationResults'];

[labels,confidence] = read_txt_files(txtfilefromfdlc);

top1prediction= labels(:,1);
top2prediction= labels(:,1:2);
top3prediction= labels(:,1:3);

%Accuracy Analysis (worm accuracy and neuron accuracy)
load(annotationpath);
[num,str] = xlsread(markernamesfile,'A2:B40');
neuronkey= string(str);
for s= 1: size(neuronkey,1)
%     if isempty(find(neuronkey(s)== experiments.Neuron_head/subset))
    if isempty(find(neuronkey(s)== experiments.Neuron_head))
        neuronkey(s)="";
    end
end
if size(num,1) ~= size(str,1)
    num_missing_end= size(num,1)-size(str,1);
    for i= 1:num_missing_end
        neuronkey= [neuronkey;""];
    end
end

top1matchcount=0;
top2matchcount=0;
top3matchcount=0;
for k=1:size(experiments.node_label,1)
        if neuronkey(k,1)~=""
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
end
numcompared= size(experiments.node_label,1) - sum(neuronkey=="");
tableofaccuracy(j,1)= top1matchcount/numcompared;
tableofaccuracy(j,2)= top2matchcount/numcompared;
tableofaccuracy(j,3)= top3matchcount/numcompared;
tableofaccuracy(j,4)= numcompared;
end


function [labels,confidence] = read_txt_files(txtfilefromfdlc)
    %%% read neuron positions
    filename = txtfilefromfdlc;
    delimiter = ',';
    startRow = 1;
    formatSpec = '%s%f%s%f%s%f%s%f%s%f';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    %fclose(fileID);
    labels= [dataArray{1} dataArray{3} dataArray{5} dataArray{7} dataArray{9}];
    confidence= [dataArray{2} dataArray{4} dataArray{6} dataArray{8} dataArray{10}];
end