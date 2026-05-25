Q = -4000:4000;
k_nl = -1/50000;
heat_nonlin_sat = Q.* exp(k_nl.*abs(Q));

heat_P_E = Q.* exp(k_nl.*Q);

plot(Q, heat_nonlin_sat)
hold on
plot(Q, heat_P_E)
plot(Q,Q)