%%% function for pre-processing images before identity labelling. This
%%% function is very simialr to another version of
%%% 'preprocess_landmark_data.m'. This script is to be used for segmenting
%%% individual stacks of worms whereas the other script can be used for
%%% preparing files for annotation whole brain videos after tracking.
%%% Performs - 1. segmentation of all channels
%%%            2. specify landmark information
%%%            3. generate coordinate axes in brain
%%%            4. generate a data file for annotation step

function preprocess_data_cpu(out_folder,data_name,img_1,img_green,img_1_marker,img_1_marker_name,varargin)

    p = inputParser;
    addRequired(p,'out_folder',@ischar)
    addRequired(p,'data_name',@ischar)
    addRequired(p,'img_1',@checkImg1)
    parse(p,out_folder,data_name,img_1)

    if ~exist('img_1')
        disp("Please provide the image in which cells are to be identified. Exiting.")
        return
    end
    
    num_imgs = length(varargin);
    
    %% segment channel in which cells are to identified
    if ~isempty(img_1_marker)
        [X,Y,Z,marker_name,marker_index] = read_marker_files(img_1_marker,img_1_marker_name);
        %adjust for z-axis resolution (for Zhang lab data, x-y resolution
        %0.4um/pixel while z resolution is 0.3um/pixel)
        %Z= Z*2/0.24; %for sihoon's imaging condition (4/3 for comparison with zhang lab images)
%         X= X*4/3;
%         Y= Y*4/3;
        mu_r = [X,Y,Z];
        labeled_img_r = [];
        cmap = rand(130+1,3);
        cmap(1,:) = [0,0,0];
    else
        
    disp("Segmenting main image channel ... ")
    cmap = rand(130+1,3);
    cmap(1,:) = [0,0,0];
    full_img = img_1;
    
    % remove noise
    temp = double(full_img);
    temp_back_corr = temp;
    t_index = find(ones(size(temp,1),size(temp,2)));
    [t_x,t_y] = ind2sub([size(temp,1),size(temp,2)],t_index);
    t_data = [t_x,t_y,ones(size(t_x,1),1)];
    for z = 1:size(temp,3)
        curr = temp(:,:,z);
        coeff = (t_data'*t_data)\t_data'*curr(t_index);
        curr_back = reshape(t_data*coeff,size(curr,1),size(curr,2));
        temp_back_corr(:,:,z) = temp(:,:,z) - curr_back;
    end
    
    for z = 1:size(temp,3)
        temp_med(:,:,z) = medfilt2(temp(:,:,z));            
        temp_med_gauss(:,:,z) = imgaussfilt(temp_med(:,:,z),1);
    end
    temp_med_gauss = mat2gray(temp_med_gauss);
    curr_img = temp_med_gauss;
    
    % determine thresh parameter
    accept_thresh = 'n';
    while ~strcmp(accept_thresh,'y')
        [thresh_param,accept_thresh,index_thresh_new] = determine_thresh_param(curr_img);
    end
    
    % segment stack
    [comp_weights,mu,sigma] = em_segmentation_3d_cpu(curr_img,index_thresh_new,size(index_thresh_new,1),[],[],5,1,cmap);
    %mu(:,3)= mu(:,3)*2/0.24;% scaling for sihoon's images
    % store segmentation information
    dummy = struct();
    dummy.comp_weights = comp_weights;
    dummy.mu = mu;
    dummy.sigma = sigma;
    dummy.index_thresh = index_thresh_new;
    seg_struct.(['f',num2str(1)]) = dummy;
    
    seg_type = 3; % segmentation done on 2d image or 3d image
    segNorm1 = false(size(full_img));
    allCenters = cell(1,size(segNorm1,4));
    numObjFound = zeros(size(segNorm1,4),1);
    for i = 1:size(segNorm1,4)
        temp_img = zeros(size(segNorm1,1),size(segNorm1,2),size(segNorm1,3));
        temp_img_filt = temp_img;
        temp_img(seg_struct.(['f',num2str(i)]).index_thresh) = 1;
        for m = 1:size(seg_struct.(['f',num2str(i)]).index_thresh,1)
            [p,q,r] = ind2sub(size(temp_img),seg_struct.(['f',num2str(i)]).index_thresh(m,1));
            temp_img_filt(max(p-10,1):min(p+10,size(temp_img,1)),max(q-10,1):min(q+10,size(temp_img,2)),max(r-2,1):min(r+2,size(temp_img,3))) = 1;
        end
        index_filt = find(temp_img_filt);
        [x,y,z] = ind2sub(size(temp_img),index_filt);
        img_data = [x,y,z];
        
        mu = seg_struct.(['f',num2str(i)]).mu;
        sigma = seg_struct.(['f',num2str(i)]).sigma;
        comp_weights = seg_struct.(['f',num2str(i)]).comp_weights;
        
        likelihood = zeros(size(img_data,1),size(comp_weights,2));
        posterior = zeros(size(img_data,1),size(comp_weights,2));
        prob_cutoff = zeros(size(img_data,1),size(comp_weights,2));
        for k = 1:size(comp_weights,2)
            likelihood(:,k) = mvnpdf(img_data,mu(k,:),sigma(:,:,k));
            posterior(:,k) = comp_weights(k)*mvnpdf(img_data,mu(k,:),sigma(:,:,k));
            prob_cutoff(:,k) = 1/sqrt((2*pi)^3*det(sigma(:,:,k)))*exp(-1/2*1);
        end
        posterior = posterior./repmat(sum(posterior,2),1,size(comp_weights,2));
    
        likelihood = likelihood.*(likelihood>= prob_cutoff);
        posterior = posterior.*(likelihood>= prob_cutoff);
        [max_posterior,max_comp] = max(posterior,[],2);
        img_pos = zeros(size(segNorm1(:,:,:,i)));
        img_pos(index_filt) = max_comp.*(max_posterior > 0.6);
        segNorm1(:,:,:,i) = img_pos;
        
        dummy = struct();
        for k = 1:size(comp_weights,2)
            Area = size(find(img_pos == k),1);
            Centroid = mu(k,:);
            pixelIndex = find(img_pos == k);
            [x,y,z] = ind2sub(size(img_pos),pixelIndex);
            PixelList = [x,y,z];
            dummy(k).Area = Area;
            dummy(k).Centroid = Centroid;
            dummy(k).PixelList = PixelList;
        end
        allCenters{1,i} = dummy;
        numObjFound(i,1) = size(comp_weights,2);
    end
    
    labeled_img = ones(size(segNorm1(:,:,:,:)));
    for i = 1:size(allCenters,2)
        for n = 1:size(allCenters{i},2)
            curr_pixelList = allCenters{i}(n).PixelList;
            if ~isempty(curr_pixelList)
                for k = 1:size(curr_pixelList,1)
                    labeled_img(curr_pixelList(k,1),curr_pixelList(k,2),curr_pixelList(k,3),i) = n+1;
                end
            end
        end
    end
    close all
    cmap = rand(150,3);
    cmap(1,:) = [0,0,0];
    for i = 1:size(labeled_img,4)
        name = strcat([out_folder,'/segmented_img_r.tif']);
        indexed_img_to_tiff(labeled_img(:,:,:,i),cmap,name)
    end
    mu_r = [mu(:,2),mu(:,1),mu(:,3)];
    labeled_img_r = labeled_img;
    end % end of if ~isempty(img_1_marker)

    %% segment other channels that may have landmarks
    mu_other_channels = {};
    labeled_img_other_channels = {};
    for chan = 1:num_imgs
        curr_chan = varargin{1,chan}; 
        if ~isempty(curr_chan{1,2})
            [X,Y,Z,marker_name,marker_index] = read_marker_files(curr_chan{1,2},curr_chan{1,3});
        mu = [X,Y,Z];
            mu_other_channels{1,chan} = mu;
            labeled_img_other_channels{1,chan} = [];
        else
        disp(['Segmenting image channel ',num2str(chan+1)])
        cmap = rand(130+1,3);
        cmap(1,:) = [0,0,0];
        full_img = curr_chan{1,1};

        % remove background in images
        temp = double(full_img);
        temp_back_corr = temp;
        t_index = find(ones(size(temp,1),size(temp,2)));
        [t_x,t_y] = ind2sub([size(temp,1),size(temp,2)],t_index);
        t_data = [t_x,t_y,ones(size(t_x,1),1)];
        for z = 1:size(temp,3)
            curr = temp(:,:,z);
            coeff = (t_data'*t_data)\t_data'*curr(t_index);
            curr_back = reshape(t_data*coeff,size(curr,1),size(curr,2));
            temp_back_corr(:,:,z) = temp(:,:,z) - curr_back;
        end    
        for z = 1:size(temp,3)
            temp_med(:,:,z) = medfilt2(temp(:,:,z));            
            temp_med_gauss(:,:,z) = imgaussfilt(temp_med(:,:,z),1);
%             temp_log(:,:,z) = imfilter(temp_med_gauss(:,:,z),lapGaufilt,'replicate');
        end
        temp_med_gauss = mat2gray(temp_med_gauss);
        curr_img = temp_med_gauss;
        
        % determine thresh parameter
        accept_thresh = 'n';
        while ~strcmp(accept_thresh,'y')
            [thresh_param,accept_thresh,index_thresh_new] = determine_thresh_param(curr_img);
        end
        
        % segment image stack
        [comp_weights,mu,sigma] = em_segmentation_3d_gpu(curr_img,index_thresh_new,size(index_thresh_new,1),[],[],5,1,cmap);
        % store segmentation information
        dummy = struct();
        dummy.comp_weights = comp_weights;
        dummy.mu = mu;
        dummy.sigma = sigma;
        dummy.index_thresh = index_thresh_new;
        seg_struct.(['f',num2str(i)]) = dummy;
    
        seg_type = 3; % segmentation done on 2d image or 3d image
        segNorm1 = false(size(full_img));
        allCenters = cell(1,size(segNorm1,4));
        numObjFound = zeros(size(segNorm1,4),1);
        for i = 1:size(segNorm1,4)
            temp_img = zeros(size(segNorm1,1),size(segNorm1,2),size(segNorm1,3));
            temp_img_filt = temp_img;
            temp_img(seg_struct.(['f',num2str(i)]).index_thresh) = 1;
            for m = 1:size(seg_struct.(['f',num2str(i)]).index_thresh,1)
                [p,q,r] = ind2sub(size(temp_img),seg_struct.(['f',num2str(i)]).index_thresh(m,1));
                temp_img_filt(max(p-10,1):min(p+10,size(temp_img,1)),max(q-10,1):min(q+10,size(temp_img,2)),max(r-2,1):min(r+2,size(temp_img,3))) = 1;
            end
            index_filt = find(temp_img_filt);
            [x,y,z] = ind2sub(size(temp_img),index_filt);
            img_data = [x,y,z];

            mu = seg_struct.(['f',num2str(i)]).mu;
            sigma = seg_struct.(['f',num2str(i)]).sigma;
            comp_weights = seg_struct.(['f',num2str(i)]).comp_weights;

            likelihood = zeros(size(img_data,1),size(comp_weights,2));
            posterior = zeros(size(img_data,1),size(comp_weights,2));
            prob_cutoff = zeros(size(img_data,1),size(comp_weights,2));
            for k = 1:size(comp_weights,2)
                likelihood(:,k) = mvnpdf(img_data,mu(k,:),sigma(:,:,k));
                posterior(:,k) = comp_weights(k)*mvnpdf(img_data,mu(k,:),sigma(:,:,k));
                prob_cutoff(:,k) = 1/sqrt((2*pi)^3*det(sigma(:,:,k)))*exp(-1/2*1);
            end
            posterior = posterior./repmat(sum(posterior,2),1,size(comp_weights,2));

            likelihood = likelihood.*(likelihood>= prob_cutoff);
            posterior = posterior.*(likelihood>= prob_cutoff);
            [max_posterior,max_comp] = max(posterior,[],2);
            img_pos = zeros(size(segNorm1(:,:,:,i)));
            img_pos(index_filt) = max_comp.*(max_posterior > 0.6);
            segNorm1(:,:,:,i) = img_pos;

            dummy = struct();
            for k = 1:size(comp_weights,2)
                Area = size(find(img_pos == k),1);
                Centroid = mu(k,:);
                pixelIndex = find(img_pos == k);
                [x,y,z] = ind2sub(size(img_pos),pixelIndex);
                PixelList = [x,y,z];
                dummy(k).Area = Area;
                dummy(k).Centroid = Centroid;
                dummy(k).PixelList = PixelList;
            end
            allCenters{1,i} = dummy;
            numObjFound(i,1) = size(comp_weights,2);
        end

        labeled_img = ones(size(segNorm1(:,:,:,:)));
        for i = 1:size(allCenters,2)
            for n = 1:size(allCenters{i},2)
                curr_pixelList = allCenters{i}(n).PixelList;
                if ~isempty(curr_pixelList)
                    for k = 1:size(curr_pixelList,1)
                        labeled_img(curr_pixelList(k,1),curr_pixelList(k,2),curr_pixelList(k,3),i) = n+1;
                    end
                end
            end
        end
        close all
        cmap = rand(150,3);
        cmap(1,:) = [0,0,0];
        for i = 1:size(labeled_img,4)
            name = strcat([out_folder,'/segmented_img_',num2str(chan+1),'.tif']);
            indexed_img_to_tiff(labeled_img(:,:,:,i),cmap,name)
        end
        
        mu_other_channels{1,chan} = [mu(:,2),mu(:,1),mu(:,3)];
        labeled_img_other_channels{1,chan} = labeled_img;
        end
    end
    disp('Segmentation finished')

    %% manually label landmark neurons by taking user input %%%%%%%
    mu_to_map = mu_r;
    landmark_channels = []; %input("Enter which channels to use for specifying landmarks e.g [2,4] else enter blank (single quotes) - ");
    if isempty(landmark_channels)
        mu_c = [];
        landmark_names_orig = {};
        landmark_to_neuron_map_orig = [];
    else
        mu_c = [];
        landmark_names_orig = {};
        landmark_to_neuron_map_orig = [];
        for i = 1:size(landmark_channels,2)
            curr_channel = landmark_channels(1,i);
            mu_curr_channel = [];
            if curr_channel == 1
                curr_mu = mu_r;
                channel_done = 'n';
                while strcmp(channel_done,'n')
                    imshow(max(mat2gray(img_1),[],3),'border','tight')
                    caxis([0,0.3])
                    hold on
                    scatter(curr_mu(:,1),curr_mu(:,2),'.r')
                    disp("Click on a landmark cell, then enter its name on terminal e.g. 'RMEL'")
                    [x,y,button] = ginput(1);
                    close all
                    mu_curr_channel = [mu_curr_channel;x,y]; %%% check the coordinate system
                    curr_name = input("Enter name of the selected landmark e.g. 'RMEL' - ");
                    landmark_names_orig = cat(1,landmark_names_orig,curr_name);
                    channel_done = input("If done with this channel, enter 'y' - ");
                end
                dist_mat = repmat(diag(mu_curr_channel*mu_curr_channel'),1,size(mu_to_map(:,1:2),1)) + repmat(diag(mu_to_map(:,1:2)*mu_to_map(:,1:2)')',size(mu_curr_channel,1),1) - 2*mu_curr_channel*mu_to_map(:,1:2)';
                [sort_dist,sort_index] = sort(dist_mat,2);
                landmark_to_neuron_map_orig = [landmark_to_neuron_map_orig;sort_index(:,1)];
                mu_c = [mu_c;mu_r(sort_index(:,1),:)];
            else
                curr_mu = mu_other_channels{1,curr_channel-1};
                channel_done = 'n';
                curr_chan = varargin{1,curr_channel-1};
                while strcmp(channel_done,'n')
                    imshow(max(mat2gray(curr_chan{1,1}),[],3),'border','tight')
                    caxis([0,0.3])
                    hold on
                    scatter(curr_mu(:,1),curr_mu(:,2),'.r')
                    disp("Click on a landmark cell, then enter its name on terminal e.g. 'RMEL'")
                    [x,y,button] = ginput(1);
                    close all
                    mu_curr_channel = [mu_curr_channel;x,y]; %%% check the coordinate system
                    curr_name = input("Enter name of the selected landmark e.g. 'RMEL' - ");
                    landmark_names_orig = cat(1,landmark_names_orig,curr_name);
                    channel_done = input("If done with this channel, enter 'y' - ");
                end
                dist_mat = repmat(diag(mu_curr_channel*mu_curr_channel'),1,size(mu_to_map(:,1:2),1)) + repmat(diag(mu_to_map(:,1:2)*mu_to_map(:,1:2)')',size(mu_curr_channel,1),1) - 2*mu_curr_channel*mu_to_map(:,1:2)';
                [sort_dist,sort_index] = sort(dist_mat,2);
                landmark_to_neuron_map_orig = [landmark_to_neuron_map_orig;sort_index(:,1)];
                mu_c = [mu_c;mu_r(sort_index(:,1),:)];
            end
        end
    end
    
    %% include landmark information from marker files if available
    if and(~isempty(img_1_marker),~isempty(img_1_marker_name))
        [X,Y,Z,marker_name,marker_index] = read_marker_files(img_1_marker,img_1_marker_name);
        % remove duplicate marker names that were specified manually
        ind_remove = [];
        for n = 1:size(marker_name,1)
            ind_remove = [];
            if find(strcmp(marker_name{n,1},landmark_names_orig))
                ind_remove = [ind_remove;n];
            end
        end
        mu_marker = [X(marker_index),Y(marker_index),Z(marker_index)];
        mu_marker(ind_remove,:) = [];
        marker_name(ind_remove,:) = [];
        
        dist_mat = repmat(diag(mu_marker*mu_marker'),1,size(mu_r,1)) + repmat(diag(mu_r*mu_r')',size(mu_marker,1),1) - 2*mu_marker*mu_r';
        [sort_dist,sort_index] = sort(dist_mat,2);
        ind_change = find(sort_dist(:,1) > 64);
        mu_r = [mu_r;mu_marker(ind_change,:)];
        dist_mat = repmat(diag(mu_marker*mu_marker'),1,size(mu_r,1)) + repmat(diag(mu_r*mu_r')',size(mu_marker,1),1) - 2*mu_marker*mu_r';
        [sort_dist,sort_index] = sort(dist_mat,2);
        
        mu_c = [mu_c;mu_marker];
        landmark_names_orig = cat(1,landmark_names_orig,marker_name);
        landmark_to_neuron_map_orig = [landmark_to_neuron_map_orig;sort_index(:,1)];
    end
    for chan = 1:num_imgs
        curr_chan = varargin{1,chan};
        if and(~isempty(curr_chan{1,2}),~isempty(curr_chan{1,3}))
            [X,Y,Z,marker_name,marker_index] = read_marker_files(curr_chan{1,2},curr_chan{1,3});
            % remove duplicate marker names that were specified manually
            ind_remove = [];
            for n = 1:size(marker_name,1)
                ind_remove = [];
                if find(strcmp(marker_name{n,1},landmark_names_orig))
                    ind_remove = [ind_remove;n];
                end
            end
            mu_marker = [X(marker_index),Y(marker_index),Z(marker_index)];
            mu_marker(ind_remove,:) = [];
            marker_name(ind_remove,:) = [];
            
            dist_mat = repmat(diag(mu_marker*mu_marker'),1,size(mu_r,1)) + repmat(diag(mu_r*mu_r')',size(mu_marker,1),1) - 2*mu_marker*mu_r';
            [sort_dist,sort_index] = sort(dist_mat,2);
            ind_change = find(sort_dist(:,1) > 64);
            mu_r = [mu_r;mu_marker(ind_change,:)];
            dist_mat = repmat(diag(mu_marker*mu_marker'),1,size(mu_r,1)) + repmat(diag(mu_r*mu_r')',size(mu_marker,1),1) - 2*mu_marker*mu_r';
            [sort_dist,sort_index] = sort(dist_mat,2);

            mu_c = [mu_c;mu_marker];
            landmark_names_orig = cat(1,landmark_names_orig,marker_name);
            landmark_to_neuron_map_orig = [landmark_to_neuron_map_orig;sort_index(:,1)];
        end
    end
    
     %visualize_landmark_to_neuron_map(thisimage_r,thisimage_c,mu_r,mu_c,landmark_to_neuron_map_orig,landmark_names_orig)

    
    %% define axes param based on PCA (noisy segmentation to improve worm
    % axes generation on strains with too little segmented cells
    
    [mu_r_noise,PA_low,LR_low, DV_low] = get_axes_from_threshold_seg(img_green,99.85, [1,2,3], out_folder); %Change thresh_param here%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %PA, LR, DV from manually segmented neurons
    mu_r_centered = mu_r - repmat(mean(mu_r),size(mu_r,1),1);
    [coeff_2,~,~] = pca(mu_r_centered);
    axes_param = [1,2,3]; %input("Enter PCA coefficients for specifying AP, LR and DV axes e.g [1,2,3] or [1,3,2] - ");%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%change this
%     PA_old = coeff_2(:,axes_param(1,1))';
%     PA_old = PA_old/norm(PA_old);
    LR_man_seg = coeff_2(:,axes_param(1,2))';
    LR_man_seg = LR_man_seg/norm(LR_man_seg);
%     DV_old = coeff_2(:,axes_param(1,3))';
%     DV_old = DV_old/norm(DV_old);
    
    % if want to use manual segmentation as the reference for LR axes
        LR_ref= LR_man_seg;
    % if want to use high threshold segmentation as the refernce for LR
%         LR_ref= LR_high/norm(LR_high);
    
    %Use the PA from noisy segmentation to find new LR and DV vectors in the
    %direction of LR from non-noisy segmentation
    fun = @(x)-(x(1)*LR_ref(1,1) + x(2)*LR_ref(1,2) + LR_ref(1,3))^2/(x(1)^2 + x(2)^2 + 1^2);
    Aeq = PA_low(1,1:2);
    beq = -PA_low(1,3);
    % x0 = coeff(:,1);
    x0 = LR_ref(1,1:2);
    LR_from_ref = fmincon(fun,x0,[],[],Aeq,beq);   % LR axis (perperndicular to PA and in direction of LR neurons)
    LR_from_ref = [LR_from_ref,1];
    LR_from_ref = LR_from_ref/norm(LR_from_ref);

    % DV axis (perperndicular to LR and PA axis)
    A = [PA_low(1,1:2);LR_from_ref(1,1:2)];
    b = [-PA_low(1,3);-LR_from_ref(1,3)];
    DV_new = inv(A'*A)*A'*b;           
    DV_new = [DV_new',1];
    DV_new = DV_new/norm(DV_new);
    
    PA= PA_low;
    LR= LR_from_ref;
    DV= DV_new;
    %%%%%%%%% Up to this has been added for noisy segmentation
    
    %% LR and DV axes correction based on symmetry
    
    [PA_updated,LR_updated,DV_updated] = correct_axes_sym(PA, LR, DV, mu_r_centered);
    
    PA= PA_updated;
    LR= LR_updated;
    DV= DV_updated;
    
 
    %%
    axes_neurons_to_neuron_map = define_axes_specifying_neurons(img_green,mu_r_noise);
    A_neuron = axes_neurons_to_neuron_map(1,1);
    P_neuron = axes_neurons_to_neuron_map(2,1);
    L_neuron = axes_neurons_to_neuron_map(3,1);
    R_neuron = axes_neurons_to_neuron_map(4,1);
    D_neuron = axes_neurons_to_neuron_map(5,1);
    V_neuron = axes_neurons_to_neuron_map(6,1);
    mu_r_centered_noise = mu_r_noise - repmat(mean(mu_r_noise),size(mu_r_noise,1),1);
    if (mu_r_centered_noise(A_neuron,:)-mu_r_centered_noise(P_neuron,:))*PA' < 0
        PA = -PA;
    end
    if D_neuron ~= 0 && V_neuron ~= 0
        if (mu_r_centered_noise(V_neuron,:)-mu_r_centered_noise(D_neuron,:))*DV' < 0
            DV = -DV;
        end
        if cross(DV,PA)*LR' < 0
            LR = -LR;
        end
    else
        if (mu_r_centered_noise(R_neuron,:)-mu_r_centered_noise(L_neuron,:))*LR' < 0
            LR = -LR;
        end
        if cross(PA,LR)*DV' < 0
            DV = -DV;
        end
    end


% % % define axes param based on PCA priginal
% %     axes_neurons_to_neuron_map = define_axes_specifying_neurons(img_green,mu_r);
% %     mu_r_centered = mu_r - repmat(mean(mu_r),size(mu_r,1),1);
% %     [coeff,~,~] = pca(mu_r_centered);
% %     axes_param = [1,2,3]; %input("Enter PCA coefficients for specifying AP, LR and DV axes e.g [1,2,3] or [1,3,2] - ");
% %     PA = coeff(:,axes_param(1,1))';
% %     PA = PA/norm(PA);
% %     LR = coeff(:,axes_param(1,2))';
% %     LR = LR/norm(LR);
% %     DV = coeff(:,axes_param(1,3))';
% %     DV = DV/norm(DV);
% %     
% %     A_neuron = axes_neurons_to_neuron_map(1,1);
% %     P_neuron = axes_neurons_to_neuron_map(2,1);
% %     L_neuron = axes_neurons_to_neuron_map(3,1);
% %     R_neuron = axes_neurons_to_neuron_map(4,1);
% %     D_neuron = axes_neurons_to_neuron_map(5,1);
% %     V_neuron = axes_neurons_to_neuron_map(6,1);
% %     if (mu_r_centered(A_neuron,:)-mu_r_centered(P_neuron,:))*PA' < 0
% %         PA = -PA;
% %     end
% %     if D_neuron ~= 0 && V_neuron ~= 0
% %         if (mu_r_centered(V_neuron,:)-mu_r_centered(D_neuron,:))*DV' < 0
% %             DV = -DV;
% %         end
% %         if cross(DV,PA)*LR' < 0
% %             LR = -LR;
% %         end
% %     else
% %         if (mu_r_centered(R_neuron,:)-mu_r_centered(L_neuron,:))*LR' < 0
% %             LR = -LR;
% %         end
% %         if cross(PA,LR)*DV' < 0
% %             DV = -DV;
% %         end
% %     end


    
%     %% Setting a spatial boundary for where the neurons are found in the
%     % image (ignoring autofluorescence in the gut)
%     %boundaryneeded = input("Enter 'y' if you want to specify a point along the AP axis above which cells are ignored - ");
%     boundaryneeded= 'n'
%     if boundaryneeded == 'y'
%     X = mu_r_noise*PA';
%     Y = mu_r_noise*LR';
%     Z = mu_r_noise*DV';
% 
%     figure, imshow(max(mat2gray(img_green),[],3),[],'border','tight');
%     hold on
%     scatter(mu_r_noise(:,1),mu_r_noise(:,2),'.r')
%     [cursorx,cursory,cursorz] = ginput(1);
% 
%     boundary= [cursorx,cursory,cursorz]*PA'
%     ind=find(X >= boundary)
%     mu_r_noise(ind,:)=[];
%     
%     axes_neurons_to_neuron_map = define_axes_specifying_neurons(img_green,mu_r_noise);
%     A_neuron = axes_neurons_to_neuron_map(1,1);
%     P_neuron = axes_neurons_to_neuron_map(2,1);
%     L_neuron = axes_neurons_to_neuron_map(3,1);
%     R_neuron = axes_neurons_to_neuron_map(4,1);
%     D_neuron = axes_neurons_to_neuron_map(5,1);
%     V_neuron = axes_neurons_to_neuron_map(6,1);
%     mu_r_centered_noise = mu_r_noise - repmat(mean(mu_r_noise),size(mu_r_noise,1),1);
%     [coeff,~,~] = pca(mu_r_centered_noise);
%     axes_param = [1,2,3]; %input("Enter PCA coefficients for specifying AP, LR and DV axes e.g [1,2,3] or [1,3,2] - ");%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     PA = coeff(:,axes_param(1,1))';
%     PA = PA/norm(PA);
%     LR = coeff(:,axes_param(1,2))';
%     LR = LR/norm(LR);
%     DV = coeff(:,axes_param(1,3))';
%     DV = DV/norm(DV);
%     
%     %Use the PA from noisy segmentation to find new LR and DV vectors in the
%     %direction of LR from non-noisy segmentation
%     fun = @(x)-(x(1)*LR_ref(1,1) + x(2)*LR_ref(1,2) + LR_ref(1,3))^2/(x(1)^2 + x(2)^2 + 1^2);
%     Aeq = PA(1,1:2);
%     beq = -PA(1,3);
%     % x0 = coeff(:,1);
%     x0 = LR_ref(1,1:2);
%     LR_from_ref = fmincon(fun,x0,[],[],Aeq,beq);   % LR axis (perperndicular to PA and in direction of LR neurons)
%     LR_from_ref = [LR_from_ref,1];
%     LR_from_ref = LR_from_ref/norm(LR_from_ref);
% 
%     % DV axis (perperndicular to LR and PA axis)
%     A = [PA(1,1:2);LR_from_ref(1,1:2)];
%     b = [-PA(1,3);-LR_from_ref(1,3)];
%     DV_new = inv(A'*A)*A'*b;           
%     DV_new = [DV_new',1];
%     DV_new = DV_new/norm(DV_new);
%     
%     LR= LR_from_ref;
%     DV= DV_new;
%     
%     if (mu_r_centered_noise(A_neuron,:)-mu_r_centered_noise(P_neuron,:))*PA' < 0
%         PA = -PA;
%     end
%     if D_neuron ~= 0 && V_neuron ~= 0
%         if (mu_r_centered_noise(V_neuron,:)-mu_r_centered_noise(D_neuron,:))*DV' < 0
%             DV = -DV;
%         end
%         if cross(DV,PA)*LR' < 0
%             LR = -LR;
%         end
%     else
%         if (mu_r_centered_noise(R_neuron,:)-mu_r_centered_noise(L_neuron,:))*LR' < 0
%             LR = -LR;
%         end
%         if cross(PA,LR)*DV' < 0
%             DV = -DV;
%         end
%     end
%     end

    %% plotting the cells and the axes
    figure, imshow(max(mat2gray(img_green),[],3),[],'border','tight');
    hold
    plot3(mu_r(:,1),mu_r(:,2),mu_r(:,3), '.r' );
    plot3([mean(mu_r(:,1)),50*PA(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),50*PA(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),50*PA(1,3)+mean(mu_r(:,3))],'b','LineWidth',1.5)
    plot3([mean(mu_r(:,1)),-20*LR(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),-20*LR(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),-20*LR(1,3)+mean(mu_r(:,3))],'g','LineWidth',1.5)
    plot3([mean(mu_r(:,1)),20*DV(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),20*DV(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),20*DV(1,3)+mean(mu_r(:,3))],'c','LineWidth',1.5)
    text(70*PA(1,1)+mean(mu_r(:,1)),70*PA(1,2)+mean(mu_r(:,2)),70*PA(1,3)+mean(mu_r(:,3)),'A','FontWeight','Bold','Color','b')
    text(-40*LR(1,1)+mean(mu_r(:,1)),-40*LR(1,2)+mean(mu_r(:,2)),-40*LR(1,3)+mean(mu_r(:,3)),'R','FontWeight','Bold','Color','g')
    text(40*DV(1,1)+mean(mu_r(:,1)),40*DV(1,2)+mean(mu_r(:,2)),40*DV(1,3)+mean(mu_r(:,3)),'V','FontWeight','Bold','Color','c')
    
    figure, imshow(max(mat2gray(img_green),[],3),[],'border','tight');
    hold
    plot3(mu_r_noise(:,1),mu_r_noise(:,2),mu_r_noise(:,3), '.r' );
    plot3([mean(mu_r_noise(:,1)),50*PA(1,1)+mean(mu_r_noise(:,1))],[mean(mu_r_noise(:,2)),50*PA(1,2)+mean(mu_r_noise(:,2))], [mean(mu_r_noise(:,3)),50*PA(1,3)+mean(mu_r_noise(:,3))],'b','LineWidth',1.5)
    plot3([mean(mu_r_noise(:,1)),-20*LR(1,1)+mean(mu_r_noise(:,1))],[mean(mu_r_noise(:,2)),-20*LR(1,2)+mean(mu_r_noise(:,2))], [mean(mu_r_noise(:,3)),-20*LR(1,3)+mean(mu_r_noise(:,3))],'g','LineWidth',1.5)
    plot3([mean(mu_r_noise(:,1)),20*DV(1,1)+mean(mu_r_noise(:,1))],[mean(mu_r_noise(:,2)),20*DV(1,2)+mean(mu_r_noise(:,2))], [mean(mu_r_noise(:,3)),20*DV(1,3)+mean(mu_r_noise(:,3))],'c','LineWidth',1.5)
    text(70*PA(1,1)+mean(mu_r_noise(:,1)),70*PA(1,2)+mean(mu_r_noise(:,2)),70*PA(1,3)+mean(mu_r_noise(:,3)),'A','FontWeight','Bold','Color','b')
    text(-40*LR(1,1)+mean(mu_r_noise(:,1)),-40*LR(1,2)+mean(mu_r_noise(:,2)),-40*LR(1,3)+mean(mu_r_noise(:,3)),'R','FontWeight','Bold','Color','g')
    text(40*DV(1,1)+mean(mu_r_noise(:,1)),40*DV(1,2)+mean(mu_r_noise(:,2)),40*DV(1,3)+mean(mu_r_noise(:,3)),'V','FontWeight','Bold','Color','c')
    
%     figure
%     plot3(mu_r_centered(:,1),mu_r_centered(:,2),mu_r_centered(:,3), '.r' );
%     hold on
%     plot3([0,50*PA(1,1)],[0,50*PA(1,2)],[0,50*PA(1,3)],'b','LineWidth',2.5)
%     plot3([0,20*LR(1,1)],[0,20*LR(1,2)],[0,20*LR(1,3)],'g','LineWidth',2.5)
%     plot3([0,20*DV(1,1)],[0,20*DV(1,2)],[0,20*DV(1,3)],'k','LineWidth',2.5)
%     text(60*PA(1,1),60*PA(1,2),60*PA(1,3),'A','FontWeight','Bold','Color','b')
%     text(22*LR(1,1),22*LR(1,2),22*LR(1,3),'R','FontWeight','Bold','Color','g')
%     text(22*DV(1,1),22*DV(1,2),22*DV(1,3),'V','FontWeight','Bold','Color','k')

    %% set axes defining parameters and save data
    ind_PCA = 1; %%% set to 1 if want to generate axes using PCA else set 0
    specify_PA = 0; %%% set to 1 if for non-pca axes generation - want to define PA axis using PCA
    landmark_names = landmark_names_orig;
    landmark_to_neuron_map = landmark_to_neuron_map_orig;
    
    if exist('SegNorm1') == 0
        segNorm1= [];
    end
    
    save([out_folder,'/',data_name],'mu_r','mu_other_channels','labeled_img_r','labeled_img_other_channels','landmark_names','landmark_to_neuron_map','mu_c','img_1','varargin','cmap','ind_PCA','specify_PA','segNorm1','PA','DV','LR')
    disp("Preprocessing finished")
end

function TF = checkImg1(x)
    TF = false;
    if isempty(x)
        error("Please provide image in which cells are to be identified")
    else
        TF = true;
    end
end

function [mu_r_noise,PA,LR, DV] = get_axes_from_threshold_seg(img_green,thresh_param, axes_param,out_folder)
    disp("Segmenting main image channel ... ")
    cmap = rand(130+1,3);
    cmap(1,:) = [0,0,0];
    full_img = img_green;
    
    % remove noise
    temp = double(full_img);
    temp_back_corr = temp;
    t_index = find(ones(size(temp,1),size(temp,2)));
    [t_x,t_y] = ind2sub([size(temp,1),size(temp,2)],t_index);
    t_data = [t_x,t_y,ones(size(t_x,1),1)];
    for z = 1:size(temp,3)
        curr = temp(:,:,z);
        coeff = (t_data'*t_data)\t_data'*curr(t_index);
        curr_back = reshape(t_data*coeff,size(curr,1),size(curr,2));
        temp_back_corr(:,:,z) = temp(:,:,z) - curr_back;
    end
    
    for z = 1:size(temp,3)
        temp_med(:,:,z) = medfilt2(temp(:,:,z));            
        temp_med_gauss(:,:,z) = imgaussfilt(temp_med(:,:,z),1);
    end
    temp_med_gauss = mat2gray(temp_med_gauss);
    curr_img = temp_med_gauss;
   
    loc_max = imregionalmax(curr_img);
    index = find(loc_max);
    [x,y,z] = ind2sub(size(loc_max),index);
    peaks = curr_img(index);
   
    thresh_val = prctile(peaks,thresh_param);
    peak_thresh =  peaks(peaks>thresh_val);
    x_thresh = x(peaks>thresh_val);
    y_thresh = y(peaks>thresh_val);
    z_thresh = z(peaks>thresh_val);
    mu_r_noise= [y_thresh,x_thresh, z_thresh];
    
    mu_r_centered_noise = mu_r_noise - repmat(mean(mu_r_noise),size(mu_r_noise,1),1);
    [coeff,~,~] = pca(mu_r_centered_noise);
    %axes_param = [1,2,3]; %input("Enter PCA coefficients for specifying AP, LR and DV axes e.g [1,2,3] or [1,3,2] - ");%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PA = coeff(:,axes_param(1,1))';
    PA = PA/norm(PA);
    LR = coeff(:,axes_param(1,2))';
    LR = LR/norm(LR);
    DV = coeff(:,axes_param(1,3))';
    DV = DV/norm(DV);
end