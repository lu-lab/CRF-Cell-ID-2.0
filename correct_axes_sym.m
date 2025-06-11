function [PA_updated,LR_updated,DV_updated] = correct_axes_sym(PA_old, LR_old, DV_old, mu_r_centered)
    
% Searching range based on each normal vectors in degree.
alpha = 20;

% Angel increment.
alpha_increment = 25;% was 1

% Tolerance for vectors calculation.
tolerance = 0.1;

% For Normal to PA.
error_normal = 0.05;

% Determines how dense the mesh will be.
discrete_level = 0.01;

% Symmetry level evaluation criteria.
threshold = 5;

% Based on DV.
best_DV = DV_old;
[symmetry_score, ~] = symmetry_level(mu_r_centered, DV_old(1), DV_old(2), DV_old(3), threshold);
reference_score_DV = symmetry_score;

vectors_DV = vector_search(DV_old(1), DV_old(2), DV_old(3), alpha, ...
    alpha_increment, discrete_level, tolerance, error_normal, PA_old);
for i = 1:size(vectors_DV, 1)
    [symmetry_score, ~] = symmetry_level(mu_r_centered, ...
        vectors_DV(i, 1), vectors_DV(i, 2), vectors_DV(i, 3), threshold);
    
    if symmetry_score < reference_score_DV
        best_DV = vectors_DV(i, :);
        reference_score_DV = symmetry_score;
    end
end

% Based on LR.
best_LR = LR_old;
[symmetry_score, ~] = symmetry_level(mu_r_centered, LR_old(1), LR_old(2), LR_old(3), threshold);
reference_score_LR = symmetry_score;

vectors_LR = vector_search(LR_old(1), LR_old(2), LR_old(3), alpha, ...
    alpha_increment, discrete_level, tolerance, error_normal, PA_old);
for i = 1:size(vectors_LR, 1)
    [symmetry_score, ~] = symmetry_level(mu_r_centered, ...
        vectors_LR(i, 1), vectors_LR(i, 2), vectors_LR(i, 3), threshold);
    
    if symmetry_score < reference_score_LR
        best_LR = vectors_LR(i, :);
        reference_score_LR = symmetry_score;
    end       
end

if reference_score_DV < reference_score_LR
    best_vector = best_DV;
else
    best_vector = best_LR;
end

PA_updated = PA_old - sum(PA_old .* best_vector) / ...
    sqrt(sum(best_vector .* best_vector)) * best_vector;
PA_updated = PA_updated / sqrt(sum(PA_updated .* PA_updated));
if PA_updated(1) * PA_old(1) < 0
    PA_updated = - PA_updated;
end
    
best_vector = best_vector / sqrt(sum(best_vector .* best_vector));

if vector_angle(best_vector, DV_old) < vector_angle(best_vector, LR_old)
    if DV_old(1) * best_vector(1) < 0
        DV_updated = -best_vector;
    else
        DV_updated = best_vector;
    end
    
    LR_updated = cross(PA_updated, DV_updated);
    LR_updated = LR_updated / norm(LR_updated);
    if LR_updated(1) * LR_old(1) < 0
        LR_updated = -LR_updated;
    end
else
    if LR_old(1) * best_vector(1) < 0
        LR_updated = -best_vector;
    else
        LR_updated = best_vector;
    end
    
    DV_updated = cross(PA_updated, LR_updated);
    DV_updated = DV_updated / norm(DV_updated);
    if DV_updated(1) * DV_old(1) < 0
        DV_updated = -DV_updated;
    end
end
end


% Return normalized vectors with a specified angle to the normal vector.
% (N1, N2, N3) is the normal vector.
function vectors = vector_search(N1, N2, N3, angle, ...
    angle_increment, discrete_level, tolerance, error_normal, PA)

% Store all vectors that satisfy the requirments.
vectors = zeros(round(angle / angle_increment) * round(1 / discrete_level^2) * 2, 3);
vector_counter = 1;

% Normalize the (N1, N2, N3) vector.
N1 = N1 / sqrt(N1^2 + N2^2 + N3^2);
N2 = N2 / sqrt(N1^2 + N2^2 + N3^2);
N3 = N3 / sqrt(N1^2 + N2^2 + N3^2);

for i = 1:round(angle / angle_increment)
    theta = i * angle_increment;
    
    % x^2 + y^2 + z^2 = 1
    % N1 * x + N2 * y + N3 * z - cosd(theat) = 0
    [x, y] = meshgrid(-1:discrete_level:1);
    x = reshape(x, [size(x, 1)^2, 1]);
    y = reshape(y, [size(y, 1)^2, 1]);
    
    for j = 1:size(x, 1)
        if 1 - x(j)^2 - y(j)^2 < 0
            continue;
        end
        
        z1 = sqrt(1 - x(j)^2 - y(j)^2);
        z2 = -sqrt(1 - x(j)^2 - y(j)^2);
        
        % Normal to PA.
        if abs(PA(1) * x(j) + PA(2) * y(j) + PA(3) * z1) > error_normal && ...
                abs(PA(1) * x(j) + PA(2) * y(j) + PA(3) * z1) > error_normal
            continue
        end
                
        if abs(N1 * x(j) + N2 * y(j) + N3 * z1 - cosd(theta)) < tolerance
            vectors(vector_counter, :) = [x(j), y(j), z1];
            vector_counter = vector_counter + 1;
        end
        if abs(N1 * x(j) + N2 * y(j) + N3 * z2 - cosd(theta)) < tolerance
            vectors(vector_counter, :) = [x(j), y(j), z2];
            vector_counter = vector_counter + 1;
        end
    end
end

% Return all vectors satisfying the requirements.
vectors = vectors(1:(vector_counter - 1), :);

end


% Return a minimum distance based on the normal vector (N1, N2, N3),
function [symmetry_score, symmetry_data] = symmetry_level(centered_data, N1, N2, N3, threshold)

% Minimum distance for all data points.
mini_distance = zeros(size(centered_data, 1), 1);

for i = 1:size(mini_distance, 1)  
    % Calculate summetry point.
    X0 = centered_data(i, 1);
    Y0 = centered_data(i, 2);
    Z0 = centered_data(i, 3);
   
    Z_mid = ((N1 * N1 + N2 * N2) * Z0 - N1 * N3 * X0 - N2 * N3 * Y0) / ...
        (N1 * N1 + N2 * N2 + N3 * N3);
    X_mid = X0 + N1 * (Z_mid - Z0) / N3;
    Y_mid = Y0 + N2 * (Z_mid - Z0) / N3;
    
    symmetry_point = [2 * X_mid - X0, 2 * Y_mid - Y0, 2 * Z_mid - Z0];
    
    % Calcualte the distance between the symmetry point and all points.
    % Find and return a minimum distance.
    minimum = 1e9;
    for j = 1:size(mini_distance, 1)
        % To determine if a point is on the same side as symmetry point.
        sign = N1 * symmetry_point(1) + N2 * symmetry_point(2) + N3 * symmetry_point(3);
        sign_current = N1 * centered_data(j, 1) + N2 * centered_data(j, 2) + ...
            N3 * centered_data(j, 3);
        if sign * sign_current < 0
            continue
        end
        
        distance = sqrt((centered_data(j, 1) - symmetry_point(1))^2 + ...
            (centered_data(j, 2) - symmetry_point(2))^2 + ...
            (centered_data(j, 3) - symmetry_point(3))^2);
        
        % Find the minimum distance.
        if distance < minimum
            minimum = distance;
        end
    end
    
    % Update the minimum distance array.
    mini_distance(i) = minimum;   
end

% Return the symmetry level based on square root summation.
symmetry_data = mini_distance;
sorted_data = sort(mini_distance);
symmetry_score = sum(sorted_data(1:threshold)) / threshold;
end

% Calculate the angle between two lines in degree.
function angle = vector_angle(V1, V2)
    V1 = reshape(V1, [3, 1]);
    V2 = reshape(V2, [3, 1]);
    angle = acosd(sum(V1 .* V2) / sqrt(sum(V1 .* V1)) / sqrt(sum(V2 .* V2)));
    if angle > 90
        angle = 180 - angle;
    end
end
