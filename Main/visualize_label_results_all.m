%%%%% visualize label results


function visualize_label_results_all(thisimage_r,mu_r,Neuron_head,node_label)
    figure, imshow(max(mat2gray(thisimage_r),[],3),[])
    caxis([0,0.3])
    hold on
    for i = 1:size(node_label,1)
        plot(mu_r(i,1),mu_r(i,2),'.r')
        if node_label(i,1)==0
        else
        text(mu_r(i,1)+2,mu_r(i,2)+2, Neuron_head{node_label(i,1),1},'Color','r','FontSize',10','Position',[-1,1,0]);
        end
    end
end