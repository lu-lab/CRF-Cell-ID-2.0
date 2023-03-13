%%%% function to updata relative position relationships based on annotated

close all
clear
clc

for iter= 1:1
%wormlist= ['20'];
%wormlist= ['01';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'24';'25';'26'];
% %wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';'25';'26';'27';'28'];
% %wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';'25';'26';'27';'28';'01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21'];
% %wormlist= ['glr1\worm01'; 'glr1\worm02'; 'glr1\worm03'; 'glr1\worm04'; 'glr1\worm05'; 'glr1\worm06'; 'glr1\worm07'; 'glr1\worm08'; 'glr1\worm09'; 'glr1\worm10'; 'glr1\worm11'; 'glr1\worm12'; 'glr1\worm13'; 'glr1\worm14'; 'glr1\worm15'; 'glr1\worm16'; 'glr1\worm17'; 'glr1\worm18'; 'glr1\worm19'; 'glr1\worm20'; 'glr1\worm21'; 'glr1\worm22'; 'glr1\worm23'; 'glr1\worm24'; 'glr1\worm25'; 'glr1\worm26'; 'glr1\worm27'; 'glr1\worm28';... 
%     %'acr5\worm01';'acr5\worm02';'acr5\worm03';'acr5\worm04';'acr5\worm05';'acr5\worm06';'acr5\worm07';'acr5\worm08';'acr5\worm09';'acr5\worm10';'acr5\worm11';'acr5\worm12';'acr5\worm13';'acr5\worm14';'acr5\worm15';'acr5\worm16';'acr5\worm17';'acr5\worm18';'acr5\worm19';'acr5\worm20';'acr5\worm21'];
% wormlist= ['27'];
wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21'];
% %wormlist= ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11'];

% %To just have 1 worm at a time
% onlyworm= wormlist(iter,:);

%To exclude 1 worm at a time
exclude_worm= wormlist(iter,:);

%To include all worms
%wormlist(iter,:)=[]; % comment this out to include all worms in the atlas and then remove exclude_worm from the out_direc name

% %%%%%%%%%%%%%% load data_neuron_relationship.mat
load('/Users/hyunjee/Dropbox (GaTech)/Whole-brain codes/CRF_Cell_ID-master/utils/data_neuron_relationship.mat')

in_direc= cell(1, size(wormlist,1)); %cell(1, 1);
whichworm_list= cell(1, size(wormlist,1));
atlasname=[];

%To exclude 1 worm at a time
for k= 1:size(wormlist,1)

% %To have just 1 worm
% for k= 1:size(onlyworm,1)
    
% %To have 1 worm at a time
% whichworm= wormlist(iter,:);

% To exclude 1 worm at a time
whichworm= wormlist(k,:);

in_direc{1,k}= ['/Users/hyunjee/Dropbox (GaTech)/Whole-brain codes/CRF_Cell_ID-master/Images/inx4mbr1/worm' whichworm];
whichworm_list{1,k}= whichworm;
atlasname=[atlasname whichworm];
end

% To have 1 worm at a time
% out_direc= ['D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\utils\data_neuron_relationship_doubleccheck_neuropal'];%neuroapl_plus_glr1_27_redo'];% onlyworm];
% 
% To exclude 1 worm at a time
out_direc= ['/Users/hyunjee/Dropbox (GaTech)/Whole-brain codes/CRF_Cell_ID-master/utils/data_neuron_relationship_rev_axes_inx4mbr1_all_'];% exclude_worm];

compiled_data = compile_annotated_data(in_direc,whichworm_list);


%%%%%%%%%%%%%% create average positional relationship matrices based on annotated data
all_annotated_neurons = {};
for i = 1:size(compiled_data,2)
    all_annotated_neurons = cat(1,all_annotated_neurons,compiled_data(i).marker_name);
end
uniq_annotated_neurons = unique(all_annotated_neurons);

%%%% remove annotated neurons that are present in 2D atlas. This is because
%%%% we do not have data for their true positional relationships
rem_index = [];
for i = 1:size(uniq_annotated_neurons,1)
    if isempty(find(strcmp(uniq_annotated_neurons{i,1},Neuron_head)))
        rem_index = [rem_index;i];
    end
end
uniq_annotated_neurons(rem_index,:) = [];

PA_consistency_matrix = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
DV_consistency_matrix = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
LR_consistency_matrix = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
%%% count matrices to keep track of in how many data sets neuron pair
%%% was observed
PA_consistency_count = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
DV_consistency_count = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
LR_consistency_count = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));

% for angle feature calculation based on annotated data, we are going to
% store an array of p-q vectors for all annotated neurons
angle_vec_data = zeros(size(uniq_annotated_neurons,1)*size(uniq_annotated_neurons,1),3);
angle_count_data = zeros(size(uniq_annotated_neurons,1)*size(uniq_annotated_neurons,1),1);
cnt = 1;
for m = 1:size(uniq_annotated_neurons,1)
    neuron_1 = uniq_annotated_neurons{m,1};
    for n = m+1:size(uniq_annotated_neurons,1)
        neuron_2 = uniq_annotated_neurons{n,1};
        for d = 1:size(compiled_data,2)
            neuron_1_index = compiled_data(d).marker_index(find(strcmp(neuron_1,compiled_data(d).marker_name)),1);
            neuron_2_index = compiled_data(d).marker_index(find(strcmp(neuron_2,compiled_data(d).marker_name)),1);
            if ~isempty(neuron_1_index) && ~isempty(neuron_2_index)
                if compiled_data(d).X_rot(neuron_1_index) < compiled_data(d).X_rot(neuron_2_index)
                    PA_consistency_matrix(m,n) = PA_consistency_matrix(m,n) + 1;
                else
                    PA_consistency_matrix(n,m) = PA_consistency_matrix(n,m) + 1;
                end
                PA_consistency_count(m,n) = PA_consistency_count(m,n) + 1;
                PA_consistency_count(n,m) = PA_consistency_count(n,m) + 1;
                
                if compiled_data(d).Y_rot(neuron_1_index) < compiled_data(d).Y_rot(neuron_2_index)
                    LR_consistency_matrix(m,n) = LR_consistency_matrix(m,n) + 1;
                else
                    LR_consistency_matrix(n,m) = LR_consistency_matrix(n,m) + 1;
                end
                LR_consistency_count(m,n) = LR_consistency_count(m,n) + 1;
                LR_consistency_count(n,m) = LR_consistency_count(n,m) + 1;
                
                if compiled_data(d).Z_rot(neuron_1_index) < compiled_data(d).Z_rot(neuron_2_index)
                    DV_consistency_matrix(m,n) = DV_consistency_matrix(m,n) + 1;
                else
                    DV_consistency_matrix(n,m) = DV_consistency_matrix(n,m) + 1;
                end
                DV_consistency_count(m,n) = DV_consistency_count(m,n) + 1;
                DV_consistency_count(n,m) = DV_consistency_count(n,m) + 1;
            end
        end
    end
end
for m = 1:size(uniq_annotated_neurons,1)
    neuron_1 = uniq_annotated_neurons{m,1};
    for n = 1:size(uniq_annotated_neurons,1)
        neuron_2 = uniq_annotated_neurons{n,1};
        for d = 1:size(compiled_data,2)
            neuron_1_index = compiled_data(d).marker_index(find(strcmp(neuron_1,compiled_data(d).marker_name)),1);
            neuron_2_index = compiled_data(d).marker_index(find(strcmp(neuron_2,compiled_data(d).marker_name)),1);
            if ~isempty(neuron_1_index) && ~isempty(neuron_2_index)
                p = [compiled_data(d).X_rot(neuron_1_index),compiled_data(d).Y_rot(neuron_1_index),compiled_data(d).Z_rot(neuron_1_index)];
                q = [compiled_data(d).X_rot(neuron_2_index),compiled_data(d).Y_rot(neuron_2_index),compiled_data(d).Z_rot(neuron_2_index)];
                ange_vec_data_index = sub2ind([size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1)],m,n);
                angle_vec_data(ange_vec_data_index,:) = angle_vec_data(ange_vec_data_index,:) + [p - q]/norm([p-q]);
                angle_count_data(ange_vec_data_index,1) = angle_count_data(ange_vec_data_index,1) + 1;
            end
        end
    end
end
mean_PA_consistency_matrix = PA_consistency_matrix./PA_consistency_count;
mean_LR_consistency_matrix = LR_consistency_matrix./LR_consistency_count;
mean_DV_consistency_matrix = DV_consistency_matrix./DV_consistency_count;
mean_angle_vec_data = angle_vec_data./repmat(angle_count_data,1,3);

%calculate standard deviation for each neuron
PA_stdev = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
DV_stdev = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
LR_stdev = zeros(size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1));
for i = 1:size(uniq_annotated_neurons,1)
    for l = 1:size(uniq_annotated_neurons,1)
        num_zero_PA= PA_consistency_count(1,2)-PA_consistency_matrix(1,2);
        PA_stdev(i,l)= sqrt((1/(PA_consistency_count(i,l)-1))*(((PA_consistency_matrix(i,l)*((1-mean_PA_consistency_matrix(i,l))^2)))+((num_zero_PA*((0-mean_PA_consistency_matrix(i,l))^2)))));
        num_zero_DV= DV_consistency_count(1,2)-DV_consistency_matrix(1,2);
        DV_stdev(i,l)= sqrt((1/(DV_consistency_count(i,l)-1))*(((DV_consistency_matrix(i,l)*((1-mean_DV_consistency_matrix(i,l))^2)))+((num_zero_DV*((0-mean_DV_consistency_matrix(i,l))^2)))));
        num_zero_LR= LR_consistency_count(1,2)-LR_consistency_matrix(1,2);
        LR_stdev(i,l)= sqrt((1/(LR_consistency_count(i,l)-1))*(((LR_consistency_matrix(i,l)*((1-mean_LR_consistency_matrix(i,l))^2)))+((num_zero_LR*((0-mean_LR_consistency_matrix(i,l))^2)))));
    end
end

%%% fill average PA, LR, DV information from data into PA, LR, DV matrices
PA_matrix_data = PA_matrix;
LR_matrix_data = LR_matrix;
DV_matrix_data = DV_matrix;
PA_matrix_stdev=  NaN(size(PA_matrix));
DV_matrix_stdev=  NaN(size(PA_matrix));
LR_matrix_stdev=  NaN(size(PA_matrix));
angle_vec_atlas = zeros(size(Neuron_head,1)*size(Neuron_head,1),3);
cnt = 1;
for m = 1:size(Neuron_head,1)
    neuron_1 = find(strcmp(Neuron_head{m,1},uniq_annotated_neurons));
    for n = 1:size(Neuron_head,1)
        neuron_2 = find(strcmp(Neuron_head{n,1},uniq_annotated_neurons));
        angle_vec_atlas_index = sub2ind([size(Neuron_head,1),size(Neuron_head,1)],m,n); 
        if ~isempty(neuron_1) && ~isempty(neuron_2)
            angle_vec_data_index = sub2ind([size(uniq_annotated_neurons,1),size(uniq_annotated_neurons,1)],neuron_1,neuron_2);
            if isnan(mean_PA_consistency_matrix(neuron_1,neuron_2))
            else
                PA_matrix_data(m,n) = mean_PA_consistency_matrix(neuron_1,neuron_2);
            end
            
            if isnan(mean_LR_consistency_matrix(neuron_1,neuron_2))
            else
                LR_matrix_data(m,n) = mean_LR_consistency_matrix(neuron_1,neuron_2);
            end
            
            if isnan(mean_DV_consistency_matrix(neuron_1,neuron_2))
            else
                DV_matrix_data(m,n) = mean_DV_consistency_matrix(neuron_1,neuron_2);
            end
            
            %%%%%%%%%%%%%added for stdev%%%%%%%%%%%%%%%%%%%%%%%%%%
            if isnan(PA_stdev(neuron_1,neuron_2))
            else
                PA_matrix_stdev(m,n) = PA_stdev(neuron_1,neuron_2);
            end
            if isnan(DV_stdev(neuron_1,neuron_2))
            else
                DV_matrix_stdev(m,n) = DV_stdev(neuron_1,neuron_2);
             end
            if isnan(LR_stdev(neuron_1,neuron_2))
            else
                LR_matrix_stdev(m,n) = LR_stdev(neuron_1,neuron_2);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if isnan(mean_angle_vec_data(angle_vec_data_index,1))
                p = [X_rot(m,1),Y_rot(m,1),Z_rot(m,1)];
                q = [X_rot(n,1),Y_rot(n,1),Z_rot(n,1)]; 
                angle_vec_atlas(angle_vec_atlas_index,:) = [p-q]/norm([p-q]);
            else
                angle_vec_atlas(angle_vec_atlas_index,:) = mean_angle_vec_data(angle_vec_data_index,:);
            end
        else
            p = [X_rot(m,1),Y_rot(m,1),Z_rot(m,1)];
            q = [X_rot(n,1),Y_rot(n,1),Z_rot(n,1)]; 
            angle_vec_atlas(angle_vec_atlas_index,:) = [p-q]/norm([p-q]);
        end
    end
end

%%% save new atlas file based on annotated data
DV_matrix = DV_matrix_data;
LR_matrix = LR_matrix_data;
PA_matrix = PA_matrix_data;
save(out_direc,'DV_matrix','eigval','eigvec','ganglion','geo_dist','LR_matrix','LR_neurons','LR_neurons_matrix','Neuron_head','PA_matrix','VC_neurons','X_rot','X_rot_norm','Y_rot','Z_rot','angle_vec_atlas')
end
