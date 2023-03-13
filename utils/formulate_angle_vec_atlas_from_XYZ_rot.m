load('random_var_for_angle_vec_correction')

angle_vec_atlas_174= zeros(174*174,1);

for i= 1:174
    for j= 1:174
    angle_vec_atlas_174(j+ 174*(i-1),1:3)= angle_vec_atlas(k_saved_178(j)+ 178*(k_saved_178(i)-1),1:3);
    % angle_vec_atlas_174(j+ 174*(i-1),1:3)= [X_rot(k_saved_178(j)) Y_rot(k_saved_178(j)) Z_rot(k_saved_178(j))] - [X_rot(k_saved_178(i)) Y_rot(k_saved_178(i)) Z_rot(k_saved_178(i))];
    end
end

