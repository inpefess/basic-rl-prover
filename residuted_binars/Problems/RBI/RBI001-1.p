include('Axioms/RBI001-0.ax').
% \^
cnf(undr_meet, hypothesis, undr(x, meet(y, z)) != meet(undr(x, y), undr(x, z))).
