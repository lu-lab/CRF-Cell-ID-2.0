partial= angle_vec_atlas(:,1);
partial_matrix= reshape(partial, 178,178);

keep_list = {'AVAL', 'AVAR', 'AVEL', 'AVER', 'RIML', 'RIMR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR',...
             'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AIBL', 'AIBR', 'URYVL', 'URYVR',...
             'AVG', 'RIGL', 'RIGR', 'RMGL','AVJR','AVJL','AIAL', 'AIAR','URYDR', 'URYDL','RMEL','RMER', 'RIAL', 'RIAR'};

keep_index = [];
for l = 1:size(keep_list,2)
    if ~isempty(find(strcmp(keep_list{1,l},Neuron_head)))%experiments.keep_list or Neuron_head or experiments.Neuron_head
        keep_index = [keep_index;find(strcmp(keep_list{1,l},Neuron_head))];
    else
        disp(keep_list{1,l})
    end
end


keep_index= sort(keep_index,'ascend');

partial_matrix_keep= partial_matrix(keep_index, keep_index);

angle_diff= partial_matrix_keep - partial_matrix_keep_ow;
howmany=sum(sum(angle_diff==0));