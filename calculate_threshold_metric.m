function [percent_correct] = calculate_threshold_metric(output, target, threshold_val)
    d1 = output./target;
    d2 = target./output;
    max_d1_d2 = max(d1,d2);
    count_mat = zeros(size(output));
    count_mat(max_d1_d2 < threshold_val) = 1;
    percent_correct = mean(count_mat,'all');
end