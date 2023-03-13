%%%%%% NeuroPAL validation
%%%%% function to compile all NeuroPAL annotated data
%%%%% 'compiled_data' variable can be used to compare variation in position
%%%%% of neurons marked in NeuroPAL strains and relative positions of those neurons

function compiled_data = compile_annotated_data_orig()
in_direc = {'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\1',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\2',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\3',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\4',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\5',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\6',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\7',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\8',...
    'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Datasets\NeuroPAL\9'}    

    compiled_data = struct();

    for i = 1:9
        
        if i == 10
            in_direc= 'D:\Dropbox (GaTech)\Whole-brain codes\CRF_Cell_ID-master\Images\glr1\worm27';
            for j= 1%:3  %j=3; %only one annotation

                if j == 1
                    annotator= 'jingting';
                    k= i;
                elseif j ==2
                    annotator= 'me';
                    k= i+size(in_direc,2); 
                elseif j ==3
                    annotator= 'myung';
                    k= i+size(in_direc,2)*2; 
                end
 
        %%% read marker files
        [X,Y,Z,marker_name,marker_index] = read_marker_files_annotation(in_direc, annotator);
        %adjust for z-axis resolution (for Zhang lab data, x-y resolution
        %2.5pixel/um while z resolution is 3.3 pixel/um)
        %Z= Z*0.75;
        %X= X*4/3;
        %Y= Y*4/3;
        
        load([in_direc '\segmentationResults_rev']);
        
%         greenImage=read_tif('one',[in_direc{1,i} '\worm' whichworm_list{1,i} 'GFP.tif']);
        
        %%%% generate axes and rotated coordinates
        mu = [X,Y,Z];
        %mu = [mu(:,2),mu(:,1),mu(:,3)];
        mu_centered = mu - repmat(mean(mu),size(mu,1),1);
        
%          %% define axes specifying neurons
%     % define axes param based on PCA (noisy segmentation to improve worm
%     % axes generation on strains with too little segmented cells
%     
%     disp("Segmenting main image channel ... ")
%     cmap = rand(130+1,3);
%     cmap(1,:) = [0,0,0];
%     full_img = greenImage;
%     
%     % remove noise
%     temp = double(full_img);
%     temp_back_corr = temp;
%     t_index = find(ones(size(temp,1),size(temp,2)));
%     [t_x,t_y] = ind2sub([size(temp,1),size(temp,2)],t_index);
%     t_data = [t_x,t_y,ones(size(t_x,1),1)];
%     for z = 1:size(temp,3)
%         curr = temp(:,:,z);
%         coeff = (t_data'*t_data)\t_data'*curr(t_index);
%         curr_back = reshape(t_data*coeff,size(curr,1),size(curr,2));
%         temp_back_corr(:,:,z) = temp(:,:,z) - curr_back;
%     end
%     
%     for z = 1:size(temp,3)
%         temp_med(:,:,z) = medfilt2(temp(:,:,z));            
%         temp_med_gauss(:,:,z) = imgaussfilt(temp_med(:,:,z),1);
%     end
%     temp_med_gauss = mat2gray(temp_med_gauss);
%     curr_img = temp_med_gauss;
%    
%     loc_max = imregionalmax(curr_img);
%     index = find(loc_max);
%     [x,y,z] = ind2sub(size(loc_max),index);
%     peaks = curr_img(index);
%     thresh_param = 99.7; %threshhold for noisy segmentation
%     thresh_val = prctile(peaks,thresh_param);
%     peak_thresh =  peaks(peaks>thresh_val);
%     x_thresh = x(peaks>thresh_val);
%     y_thresh = y(peaks>thresh_val);
%     z_thresh = z(peaks>thresh_val); 
%     index_thresh = index(peaks>thresh_val);
%     [x_thresh_new,y_thresh_new,z_thresh_new,index_thresh_new] = remove_close_neurons(x_thresh,y_thresh,z_thresh,index_thresh,curr_img);
%     figure,imshow(max(curr_img,[],3),[],'border','tight')
%     caxis([0,0.6])
%     hold on
%     scatter(y_thresh,x_thresh,'.g')
%     scatter(y_thresh_new,x_thresh_new,'.r')
%     text(30,30,['Number of detected cells - ',num2str(size(x_thresh_new,1))],'Color','white')
%     close all
%     
%     % segment stack
%     [comp_weights_noise,mu_noise,sigma_noise] = em_segmentation_3d_gpu(curr_img,index_thresh_new,size(index_thresh_new,1),[],[],5,1,cmap);
%     % store segmentation information
%     dummy = struct();
%     dummy.comp_weights = comp_weights_noise;
%     dummy.mu = mu_noise;
%     dummy.sigma = sigma_noise;
%     dummy.index_thresh = index_thresh_new;
%     seg_struct.(['f',num2str(1)]) = dummy;
%     
%     seg_type = 3; % segmentation done on 2d image or 3d image
%     segNorm1 = false(size(full_img));
%     allCenters = cell(1,size(segNorm1,4));
%     numObjFound = zeros(size(segNorm1,4),1);
%     for i = 1:size(segNorm1,4)
%         temp_img = zeros(size(segNorm1,1),size(segNorm1,2),size(segNorm1,3));
%         temp_img_filt = temp_img;
%         temp_img(seg_struct.(['f',num2str(i)]).index_thresh) = 1;
%         for m = 1:size(seg_struct.(['f',num2str(i)]).index_thresh,1)
%             [p,q,r] = ind2sub(size(temp_img),seg_struct.(['f',num2str(i)]).index_thresh(m,1));
%             temp_img_filt(max(p-10,1):min(p+10,size(temp_img,1)),max(q-10,1):min(q+10,size(temp_img,2)),max(r-2,1):min(r+2,size(temp_img,3))) = 1;
%         end
%         index_filt = find(temp_img_filt);
%         [x,y,z] = ind2sub(size(temp_img),index_filt);
%         img_data = [x,y,z];
%         
%         mu_noise = seg_struct.(['f',num2str(i)]).mu;
%         sigma_noise = seg_struct.(['f',num2str(i)]).sigma;
%         comp_weights_noise = seg_struct.(['f',num2str(i)]).comp_weights;
%         
%         likelihood = zeros(size(img_data,1),size(comp_weights_noise,2));
%         posterior = zeros(size(img_data,1),size(comp_weights_noise,2));
%         prob_cutoff = zeros(size(img_data,1),size(comp_weights_noise,2));
%         for k = 1:size(comp_weights_noise,2)
%             likelihood(:,k) = mvnpdf(img_data,mu_noise(k,:),sigma_noise(:,:,k));
%             posterior(:,k) = comp_weights_noise(k)*mvnpdf(img_data,mu_noise(k,:),sigma_noise(:,:,k));
%             prob_cutoff(:,k) = 1/sqrt((2*pi)^3*det(sigma_noise(:,:,k)))*exp(-1/2*1);
%         end
%         posterior = posterior./repmat(sum(posterior,2),1,size(comp_weights_noise,2));
%     
%         likelihood = likelihood.*(likelihood>= prob_cutoff);
%         posterior = posterior.*(likelihood>= prob_cutoff);
%         [max_posterior,max_comp] = max(posterior,[],2);
%         img_pos = zeros(size(segNorm1(:,:,:,i)));
%         img_pos(index_filt) = max_comp.*(max_posterior > 0.6);
%         segNorm1(:,:,:,i) = img_pos;
%         
%         dummy = struct();
%         for k = 1:size(comp_weights_noise,2)
%             Area = size(find(img_pos == k),1);
%             Centroid = mu_noise(k,:);
%             pixelIndex = find(img_pos == k);
%             [x,y,z] = ind2sub(size(img_pos),pixelIndex);
%             PixelList = [x,y,z];
%             dummy(k).Area = Area;
%             dummy(k).Centroid = Centroid;
%             dummy(k).PixelList = PixelList;
%         end
%         allCenters{1,i} = dummy;
%         numObjFound(i,1) = size(comp_weights_noise,2);
%     end
%     
%     labeled_img = ones(size(segNorm1(:,:,:,:)));
%     for i = 1:size(allCenters,2)
%         for n = 1:size(allCenters{i},2)
%             curr_pixelList = allCenters{i}(n).PixelList;
%             if ~isempty(curr_pixelList)
%                 for k = 1:size(curr_pixelList,1)
%                     labeled_img(curr_pixelList(k,1),curr_pixelList(k,2),curr_pixelList(k,3),i) = n+1;
%                 end
%             end
%         end
%     end
%     close all
%     cmap = rand(150,3);
%     cmap(1,:) = [0,0,0];
%     for i = 1:size(labeled_img,4)
%         name = strcat([out_folder,'\segmented_img_r.tif']);
%         indexed_img_to_tiff(labeled_img(:,:,:,i),cmap,name)
%     end
%     mu_r_noise = [mu_noise(:,2),mu_noise(:,1),mu_noise(:,3)];
%     labeled_img_r = labeled_img;
%     
%     axes_neurons_to_neuron_map = define_axes_specifying_neurons(img_1,mu_r_noise);
%     
%     mu_r_centered_noise = mu_r_noise - repmat(mean(mu_r_noise),size(mu_r_noise,1),1);
%     [coeff,~,~] = pca(mu_r_centered_noise);
%     axes_param = [1,3,2]; %input("Enter PCA coefficients for specifying AP, LR and DV axes e.g [1,2,3] or [1,3,2] - ");
%     PA = coeff(:,axes_param(1,1))';
%     PA = PA/norm(PA);
%     LR = coeff(:,axes_param(1,2))';
%     LR = LR/norm(LR);
%     DV = coeff(:,axes_param(1,3))';
%     DV = DV/norm(DV);  
% 
%     A_neuron = marker_index(find(strcmp('URADR',marker_name)),1);
%     P_neuron = marker_index(find(strcmp('ALA',marker_name)),1);
%     L_neuron = marker_index(find(strcmp('URADL',marker_name)),1);
%     R_neuron = marker_index(find(strcmp('URADR',marker_name)),1);
%     D_neuron = 0;
%     V_neuron = 0;
%     
%     if (mu_centered(A_neuron,:)-mu_centered(P_neuron,:))*PA' < 0
%         PA = -PA;
%     end
%     if D_neuron ~= 0 && V_neuron ~= 0
%         if (mu_centered(V_neuron,:)-mu_centered(D_neuron,:))*DV' < 0
%             DV = -DV;
%         end
%         if cross(DV,PA)*LR' < 0
%             LR = -LR;
%         end
%     else
%         if (mu_centered(R_neuron,:)-mu_centered(L_neuron,:))*LR' < 0
%             LR = -LR;
%         end
%         if cross(PA,LR)*DV' < 0
%             DV = -DV;
%         end
%     end
%         
%         if cross(PA,LR)*DV' < 0
%             LR = -LR;
%         end

        X_rot = mu_centered*PA';
        Y_rot = mu_centered*LR';
        Z_rot = mu_centered*DV';

% % %     visualize    
%     figure, imshow(max(mat2gray(greenImage),[],3),[],'border','tight');
%     caxis([0 0.1])
%     hold
%     plot3(mu_r(:,1),mu_r(:,2),mu_r(:,3), '.r' );
%     plot3([mean(mu_r(:,1)),50*PA(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),50*PA(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),50*PA(1,3)+mean(mu_r(:,3))],'b','LineWidth',1.5)
%     plot3([mean(mu_r(:,1)),-20*LR(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),-20*LR(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),-20*LR(1,3)+mean(mu_r(:,3))],'g','LineWidth',1.5)
%     plot3([mean(mu_r(:,1)),20*DV(1,1)+mean(mu_r(:,1))],[mean(mu_r(:,2)),20*DV(1,2)+mean(mu_r(:,2))], [mean(mu_r(:,3)),20*DV(1,3)+mean(mu_r(:,3))],'c','LineWidth',1.5)
%     text(70*PA(1,1)+mean(mu_r(:,1)),70*PA(1,2)+mean(mu_r(:,2)),70*PA(1,3)+mean(mu_r(:,3)),'A','FontWeight','Bold','Color','b')
%     text(-40*LR(1,1)+mean(mu_r(:,1)),-40*LR(1,2)+mean(mu_r(:,2)),-40*LR(1,3)+mean(mu_r(:,3)),'R','FontWeight','Bold','Color','g')
%     text(40*DV(1,1)+mean(mu_r(:,1)),40*DV(1,2)+mean(mu_r(:,2)),40*DV(1,3)+mean(mu_r(:,3)),'V','FontWeight','Bold','Color','c')
    
        
        compiled_data(i).X_rot = X_rot;
        compiled_data(i).Y_rot = Y_rot;
        compiled_data(i).Z_rot = Z_rot;
        compiled_data(i).marker_index = marker_index;
        compiled_data(i).marker_name = marker_name;
            end   
            
        else
        %%% read marker files
        annotator= 'jingting';
        [X,Y,Z,marker_name,marker_index] = read_marker_files_annotation_saved(in_direc{1,i});

        %%%% generate axes and rotated coordinates
        mu = [X,Y,Z];
        mu_centered = mu - repmat(mean(mu),size(mu,1),1);
        [coeff,score,latent] = pca(mu_centered);
        PA = coeff(:,1)';
        PA = PA/norm(PA);
        DV = coeff(:,2)';
        DV = DV/norm(DV);
        LR = coeff(:,3)';
        LR = LR/norm(LR);

        OLLL_neuron = marker_index(find(strcmp('OLLL',marker_name)),1);
        OLLR_neuron = marker_index(find(strcmp('OLLR',marker_name)),1);
        AVAL_neuron = marker_index(find(strcmp('AVAL',marker_name)),1);
        AVAR_neuron = marker_index(find(strcmp('AVAR',marker_name)),1);
        if ~isempty(OLLL_neuron)
            if ~isempty(AVAL_neuron)
                if (mu_centered(OLLL_neuron,:)-mu_centered(AVAL_neuron,:))*PA' < 0
                    PA = -PA;
                end
            else
                if (mu_centered(OLLL_neuron,:)-mu_centered(AVAR_neuron,:))*PA' < 0
                    PA = -PA;
                end
            end
        else
            if ~isempty(AVAL_neuron)
                if (mu_centered(OLLR_neuron,:)-mu_centered(AVAL_neuron,:))*PA' < 0
                    PA = -PA;
                end
            else
                if (mu_centered(OLLR_neuron,:)-mu_centered(AVAR_neuron,:))*PA' < 0
                    PA = -PA;
                end
            end
        end

        ALA_neuron = marker_index(find(strcmp('ALA',marker_name)),1);
        RMDDL_neuron = marker_index(find(strcmp('RMDDL',marker_name)),1);
        RMDDR_neuron = marker_index(find(strcmp('RMDDR',marker_name)),1);
        if ~isempty(RMDDL_neuron)
            if (mu_centered(RMDDL_neuron,:)-mu_centered(ALA_neuron,:))*DV' < 0
                DV = -DV;
            end
        else
            if (mu_centered(RMDDR_neuron,:)-mu_centered(ALA_neuron,:))*DV' < 0
                DV = -DV;
            end
        end
        if cross(PA,LR)*DV' < 0
            LR = -LR;
        end

        X_rot = mu_centered*PA';
        Y_rot = mu_centered*LR';
        Z_rot = mu_centered*DV';
        figure,scatter3(mu_centered(:,1),mu_centered(:,2),mu_centered(:,3),20,'or','filled')
        hold on
        plot3(100*[0,PA(1,1)],100*[0,PA(1,2)],100*[0,PA(1,3)],'b','LineWidth',5)
        plot3(50*[0,LR(1,1)],50*[0,LR(1,2)],30*[0,LR(1,3)],'g','LineWidth',5)
        plot3(50*[0,DV(1,1)],50*[0,DV(1,2)],50*[0,DV(1,3)],'k','LineWidth',5)
        
        %%%%keep only one RIGR neuron
        ind_RIGR = find(strcmp('RIGR',marker_name));
        if size(ind_RIGR,1) > 1
            X_rot(ind_RIGR(1,1),:) = [];
            Y_rot(ind_RIGR(1,1),:) = [];
            Z_rot(ind_RIGR(1,1),:) = [];
            marker_index(ind_RIGR(1,1),:) = [];
            marker_name(ind_RIGR(1,1),:) = [];
        end
        ind_RIGL = find(strcmp('RIGL',marker_name));
        if size(ind_RIGL,1) > 1
            X_rot(ind_RIGL(1,1),:) = [];
            Y_rot(ind_RIGL(1,1),:) = [];
            Z_rot(ind_RIGL(1,1),:) = [];
            marker_index(ind_RIGL(1,1),:) = [];
            marker_name(ind_RIGL(1,1),:) = [];
        end
        
        compiled_data(i).X_rot = X_rot;
        compiled_data(i).Y_rot = Y_rot;
        compiled_data(i).Z_rot = Z_rot;
        compiled_data(i).marker_index = marker_index;
        compiled_data(i).marker_name = marker_name;

%         rgb_img_annotated_data(in_direc{1,i},X,Y,marker_name,marker_index)
    end
    end
end