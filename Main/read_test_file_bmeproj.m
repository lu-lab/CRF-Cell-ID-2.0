%%%%%%% function to read marker and marker names file. These files specify
%%%%%%% the green channel neuron positions and names obtained by annotating
%%%%%%% in Vaa3D
%%%%%% Inputs - 1. input_2 variable specifying folder path where 'marker'
%%%%%%             and 'marker_names.xlsx' are present. Specify it in
%%%%%%             terminal not in the code since this code is used in
%%%%%%             other codes as well.

function [X,Y,Z] = read_test_file_bmeproj(img_1_marker)
    %%% read neuron positions
    filename = img_1_marker;
    delimiter = ',';
    startRow = 2;
    formatSpec = '%f%f%f%f%f%s%s%f%f%f%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    markers = table(dataArray{1:end-1}, 'VariableNames', {'x','y','z','radius','shape','name','comment','color_r','color_g','color_b'});
    X = table2array(markers(:,1));
    Y = table2array(markers(:,2));
    Z = table2array(markers(:,3));
end