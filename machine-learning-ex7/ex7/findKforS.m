function [ K ] = findKforS( S, varianceGoal)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    K = 1;
    mTotal = sum(sum(S));
    kTotal = S(K,K);
    while (K <= kTotal)
        variance = kTotal / mTotal;
%         fprintf('K = %d. Variance = %.4f.\n', K, variance);
        if (variance >= 1 - varianceGoal)
            break;
        end
        K = K+ 1;
        kTotal = kTotal + S(K,K);
    end

end

